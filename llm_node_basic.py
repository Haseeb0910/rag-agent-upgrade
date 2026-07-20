import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph , START , END
from typing import TypedDict
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
    
class GraphState(TypedDict):
    question : str
    answer : str
    
def ask_llm(state: GraphState) -> GraphState:
    print("Calling LLM...")
    response = llm.invoke(state["question"])
    state["answer"] = response.content
    return state
    

builder = StateGraph(GraphState)

builder.add_node("ask_llm" , ask_llm)

builder.add_edge(START, "ask_llm")
builder.add_edge("ask_llm" , END)

graph = builder.compile()

result = graph.invoke({"question": "what is the capital of pakistan?" , "answer" : ""})

print(result['question'])
print(result['answer'])