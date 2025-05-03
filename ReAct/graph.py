from langchain_core.agents import AgentFinish
from langgraph.graph import END,StateGraph
from nodes import execute_tools,run_agent_reasoning
from State import AgentState
from Model_LLM_temp import save_image,clear_images

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


def execute(promnt,img=None):
    if img:
        file_path=save_image(img)
        final_promnt=promnt+f"/nImage_path {file_path}"
        print(final_promnt)
        res = agent.invoke({"input": final_promnt})
        clear_images()
    else:
        res = agent.invoke(input={"input": promnt})
    images=[]
    for state in res["intermediate_steps"]:
        if "generate_photo" in state[0].tool:
            images.append(state[1])
    if len(images) == 0:
        return res["agent_outcome"].return_values["output"]
    return res["agent_outcome"].return_values["output"],images





