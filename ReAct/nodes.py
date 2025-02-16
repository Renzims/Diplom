from State import AgentState
from ReAct import ReAct_agent, tools
from langgraph.prebuilt.tool_executor import ToolExecutor

def run_agent_reasoning(state:AgentState):
    agent_outcome=ReAct_agent.invoke(state)
    return {"agent_outcome":agent_outcome}

tool_executor=ToolExecutor(tools)

def execute_tools(state:AgentState):
    agent_action = state["agent_outcome"]
    output=tool_executor.invoke(agent_action)
    return {"intermediate_steps":[(agent_action,output)]}



