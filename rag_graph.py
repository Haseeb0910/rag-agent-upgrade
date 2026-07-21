import os 
os.environ["HF_HUB_OFFLINE"] = "1"
from dotenv import load_dotenv
from langgraph.graph import StateGraph , START , END
from typing import TypedDict
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np 

load_dotenv()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

print("Loading embedding model...")
# os.environ["HF_HUB_ETAG_TIMEOUT"] = "5"
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text , chunk_size = 1000 , overlap = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

PDF_PATH = "sample.pdf"

print("Building index from document...")
raw_text = get_text_from_pdf(PDF_PATH)
chunks = split_text(raw_text)
embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
print(f"Index built with {len(chunks)} chunks.")

class GraphState(TypedDict):
    question : str
    context : str
    answer : str
    
def retrieve(state: GraphState) -> GraphState:
    print("Retrieving relevant chunks...")
    question_embedding = embedder.encode([state["question"]])
    distances, indices = index.search(np.array(question_embedding), k=3)
    retrieved_context = "\n\n".join([chunks[i] for i in indices[0]])
    state["context"] = retrieved_context
    return state
    
def ask_llm(state : GraphState) -> GraphState:
    print("Calling LLM with retrieved context...")   
    prompt = f"""Use the context below to answer the question. If the answer isn't in the context, say "I don't know."
    context : {state["context"]}
    question : {state["question"]}
    """
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state
    
builder = StateGraph(GraphState)
builder.add_node("retrieve" , retrieve)
builder.add_node("ask_llm" , ask_llm)

builder.add_edge(START , "retrieve")
builder.add_edge("retrieve" , "ask_llm")
builder.add_edge("ask_llm" , END)

graph = builder.compile()

result = graph.invoke({"question": "difference between subnetting and supernetting?", "context": "" , "answer": ""})
print("\n--- Final Answer ---")
print(result["answer"])