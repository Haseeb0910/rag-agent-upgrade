from langgraph.graph import StateGraph , START , END
from typing import TypedDict

class GraphState (TypedDict):
    input : str 
    step_log : list
    
def check_input(state : GraphState) -> GraphState:
    print("Entered check_input")
    state["step_log"].append("check_input")
    return state

def handle_greeting(state: GraphState) -> GraphState:
    print("Entered handle_greeting")
    state["step_log"].append("handle_greeting")
    return state

def handle_other(state: GraphState) -> GraphState:
    print("Entered handle_other")
    state["step_log"].append("handle_other")
    return state

def route_decision(state : GraphState) -> str:
    if "hello" in state["input"].lower():
        return "handle_greeting"
    else: 
        return "handle_other"
    
builder = StateGraph(GraphState)
builder.add_node("check_input" , check_input) 
builder.add_node("handle_greeting" , handle_greeting) 
builder.add_node("handle_other" , handle_other) 

builder.add_edge(START , "check_input")

builder.add_conditional_edges(
    "check_input" , 
    route_decision,
    {
        "handle_greeting" : "handle_greeting" ,
        "handle_other" : "handle_other" 
    }
    )
builder.add_edge("handle_greeting" , END)
builder.add_edge("handle_other" , END)

graph = builder.compile()

# Test with a greeting
response1 = graph.invoke({"input": "hello there", "step_log": []})
print("Test 1:", response1)

# Test with something else
response2 = graph.invoke({"input": "what's the weather", "step_log": []})
print("Test 2:", response2)