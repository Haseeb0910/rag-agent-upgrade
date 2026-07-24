import os
os.environ["HF_HUB_OFFLINE"] = "1"

from dotenv import load_dotenv
import gradio as gr
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage , SystemMessage
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
import numpy as np
import uuid

load_dotenv()

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# --- Global state for the currently loaded document ---
current_chunks = []
current_index = None

def build_index_from_file(file_obj):
    global current_chunks, current_index
    if file_obj is None:
        return "No file uploaded."
    
    text = get_text_from_pdf(file_obj.name)
    if not text:
        return "Could not extract text from file."
    
    current_chunks = split_text(text)
    embeddings = embedder.encode(current_chunks)
    current_index = faiss.IndexFlatL2(embeddings.shape[1])
    current_index.add(np.array(embeddings))
    return f"Document loaded — {len(current_chunks)} chunks indexed. Ask away!"

@tool
def search_doc(query: str) -> str:
    """Search the uploaded document for information relevant to the query."""
    if current_index is None:
        return "No document has been uploaded yet."
    query_embedding = embedder.encode([query])
    distances, indices = current_index.search(np.array(query_embedding), k=3)
    return "\n\n".join([current_chunks[i] for i in indices[0]])

@tool
def calculate(expression: str) -> str:
    """Evaluate a basic math expression, like '15 + 27' or '10 * 3'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

tools = [search_doc, calculate]
tool_map = {"search_doc": search_doc, "calculate": calculate}

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
llm_with_tools = llm.bind_tools(tools)

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: GraphState) -> GraphState:
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        print(f"Tool call: {response.tool_calls[0]['name']}")
    return {"messages": [response]}

def should_continue(state: GraphState) -> str:
    last_message = state["messages"][-1]
    return "call_tool" if last_message.tool_calls else "end"

def call_tool(state: GraphState) -> GraphState:
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_fn = tool_map[tool_call["name"]]
    result = tool_fn.invoke(tool_call["args"])
    tool_message = ToolMessage(content=result, tool_call_id=tool_call["id"])
    return {"messages": [tool_message]}

builder = StateGraph(GraphState)
builder.add_node("call_model", call_model)
builder.add_node("call_tool", call_tool)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, {"call_tool": "call_tool", "end": END})
builder.add_edge("call_tool", "call_model")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

def chat_fn(message, history, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    system_msg = SystemMessage(content="You are a helpful assistant. Only use search_doc when the question is clearly about the uploaded document's content. Only use calculate for math expressions. For general knowledge questions unrelated to the document (like facts, definitions, or trivia), answer directly from what you already know — do not attempt to search for anything and do not invent tools that don't exist. Use search_doc at most once per question.")
    try:
        result = graph.invoke(
            {"messages": [system_msg, HumanMessage(content=message)]},
            config={**config, "recursion_limit": 8}
        )
        return result["messages"][-1].content
    except Exception as e:
        print(f"Error during graph invocation: {e}")
        return "Sorry, I ran into an issue processing that. Could you try rephrasing your question?"

with gr.Blocks(title="Agentic RAG Chatbot") as demo:
    gr.Markdown("# 🤖 Agentic RAG Chatbot\nUpload a PDF, then ask questions. I can search your document or do math.")
    
    thread_id_state = gr.State(str(uuid.uuid4()))  # unique conversation ID per session
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            status_box = gr.Textbox(label="Status", interactive=False)
            file_input.upload(fn=build_index_from_file, inputs=[file_input], outputs=[status_box])
        
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[thread_id_state],
            )

if __name__ == "__main__":
    demo.launch()