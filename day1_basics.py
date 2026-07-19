from langgraph.graph import StateGraph , START , END
from typing import TypedDict

class GraphState(TypedDict):
    input : str
    step_log : list
    
# Define node functions - each just modifies state and returns it
def first_node(state : GraphState) -> GraphState:
    print("Entered node_one")
    state["step_log"].append("node_one")
    return state

def second_node(state : GraphState) -> GraphState:
    print("Entered node_two")
    state["step_log"].append("node_two")
    return state

def third_node(state : GraphState) -> GraphState:
    print("Entered node_three")
    state["step_log"].append("node_three")
    return state


# building the graph
builder = StateGraph(GraphState)
# adding nodes
builder.add_node("first_node" , first_node)
builder.add_node("second_node" , second_node)
builder.add_node("third_node" , third_node)
# adding edges
builder.add_edge(START , "first_node")
builder.add_edge("first_node" , "second_node")
builder.add_edge("second_node" , "third_node")
builder.add_edge("third_node" , END)
# compiling the graph
graph = builder.compile()
# running the grpah
response = graph.invoke({"input": "hello" , "step_log" : []})

print("Final response:" , response)

