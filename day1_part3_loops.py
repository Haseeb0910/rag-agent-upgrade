from langgraph.graph import StateGraph , START , END
from typing import TypedDict

class GraphState(TypedDict):
    attempt : int
    max_attempts : int
    success : bool 
    
def try_task(state : GraphState) -> GraphState:
    state["attempt"] += 1
    print(f"Attempt {state['attempt']}...")
    if state["attempt"] >= 3:
        state["success"] = True
    return state

def check_result (state: GraphState) -> str:
    if state["success"]:
        return "done"
    elif state["attempt"] >= state["max_attempts"]:
        print("Giving up - hit max attempts")
        return "done" 
    else:
        return "retry"
    
        
builder = StateGraph(GraphState)

builder.add_node("try_task" , try_task)

builder.add_edge(START , "try_task")

builder.add_conditional_edges(
    "try_task" , 
    check_result , 
    {
        "retry" : "try_task",
        "done" : END
    }
    )

graph = builder.compile()

response = graph.invoke({"attempt" : 0 , "max_attempts" : 5 , "success" : False})
print("Final response:", response)