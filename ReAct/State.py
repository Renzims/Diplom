from typing import Annotated, Union
from typing_extensions import TypedDict
import operator
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish

class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction,AgentFinish,None]
    intermediate_steps: Annotated[list[tuple[AgentAction,BaseMessage]],operator.add]