import os 
os.environ["HF_HUB_OFFLINE"] = "1"
from dotenv import load_dotenv
from langgraph.graph import StateGraph , START , END
from typing import TypedDict , Annotated
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np 
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

print("Loading embedding model...")
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

@tool 
def search_doc(query : str) -> str:
    """Search the uploaded document for information relevant to the query.
    Use this when the user asks something that might be answered by the document's content."""
    print(f"[TOOL CALLED] search_document('{query}')")
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=3)
    return "\n\n".join([chunks[i] for i in indices[0]])

@tool
def calculate(expression: str) -> str:
    """Evaluate a basic math expression, like '15 + 27' or '10 * 3'.
    Use this when the user asks a math question."""
    print(f"[TOOL CALLED] calculate('{expression}')")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

tools = [search_doc , calculate]

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

llm_with_tools = llm.bind_tools(tools)

class GraphState(TypedDict):
    messages : Annotated[list , add_messages]
  
def call_model(state: GraphState) -> GraphState:
    print("LLM Thinking...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages" : [response]}

def should_continue(state: GraphState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:   # did the LLM decide to call a tool?
        return "call_tool"
    return "end" 
    
tool_map = {"search_doc": search_doc, "calculate": calculate}

def call_tool(state: GraphState) -> GraphState:
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_fn = tool_map[tool_call["name"]]
    result = tool_fn.invoke(tool_call["args"])
    from langchain_core.messages import ToolMessage
    tool_message = ToolMessage(content=result, tool_call_id=tool_call["id"])
    return {"messages": [tool_message]}

    
builder = StateGraph(GraphState)
builder.add_node("call_model" , call_model)
builder.add_node("call_tool" , call_tool)

builder.add_edge(START , "call_model")
builder.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "call_tool" : "call_tool",
        "end" : END
    }
)
builder.add_edge("call_tool", "call_model")  

graph = builder.compile()

# Test 1
response1 = graph.invoke({"messages" : [HumanMessage(content="What is subnetting used for?")]})
print("\n--- Answer 1 ---")
print(response1["messages"][-1].content)

# Test 2
response2 = graph.invoke({"messages" : [HumanMessage(content="what is 15+27?")]})
print("\n--- Answer 2 ---")
print(response2["messages"][-1].content)