from langchain_core.agents import AgentFinish
from langgraph.graph import END,StateGraph
from nodes import execute_tools,run_agent_reasoning
from State import AgentState

AGENT_REASON="agent_reason"
ACT="act"

def sould_cont(state:AgentState):
    if isinstance(state["agent_outcome"],AgentFinish):
        return END
    else:
        return ACT

flow=StateGraph(AgentState)
flow.add_node(AGENT_REASON,run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)

flow.add_node(ACT,execute_tools)
flow.add_conditional_edges(
    AGENT_REASON,
    sould_cont,
)
flow.add_edge(ACT,AGENT_REASON)

agent=flow.compile()


def execute(promnt):
    res = agent.invoke(input={
        "input": promnt
    })
    images=[]
    for state in res["intermediate_steps"]:
        if "generate_photo" in state[0].tool:
            images.append(state[1])
    if len(images) == 0:
        return res["agent_outcome"].return_values["output"]
    return res["agent_outcome"].return_values["output"],images

"""
if __name__ == "__main__":
    print("hello")

    res=agent.invoke(input={
        "input":"I need to depict a dragon flying over a lake at sunset. Generate 5 different photo"
    })

    #res=execute("I need to depict a dragon flying over a lake at sunset. Generate 5 different photo")
    print(res)
"""



