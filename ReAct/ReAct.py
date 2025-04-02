from langchain import hub
from langchain.agents import create_react_agent
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from sympy.physics.units import temperature

from Model_Photo_temp import model as photo_model
from Model_LLM_temp import model, template
from PIL import Image
import json
from API import OpenAI
from history import collection

@tool
def generate_photo(prompt: str):
    """
    Call when you need to generate an image.
    Can generate only one image per call.
    """
    image = photo_model(prompt, num_inference_steps=150).images[0]
    return image



@tool
def image_analysis(prompt:str):
    """
    Call when you need to analyze images sent by the user.
    You must pass the user's request and the path to the image that was sent to you.
    Query format in the tool:
        query: {query}
        image_path: {image_path}
    """
    data = json.loads(prompt)
    query=template.format(query=data["query"])
    answer=model._call(prompt=query,image=Image.open(data["image_path"]))
    return answer

@tool
def history(prompt:str):
    """
    This tool returns the last 3 pairs of the user's query/your answer.
    You should use it to get more context in the conversation with the user or the user can refer to their previous messages.
    If the user's request is for images, use only the descriptions provided in the story.
    """
    print("USE HIST")
    documents = collection.find(
        {"role": {"$in": ["user", "assistant"]}, "content": {"$exists": True}}
    ).sort("_id", -1).limit(6)
    formatted_output = "Last 3 pairs user request/your answer\n"
    user_query = None
    for doc in reversed(documents.to_list()):
        if doc["role"] == "user":
            user_query = doc["content"]
        elif doc["role"] == "assistant" and user_query:
            formatted_output += f"User query: {user_query}\nYou answer: {doc['content']}\n\n"
            user_query = None
    return formatted_output

agent_prompt = hub.pull("hwchase17/react")
tools=[generate_photo,image_analysis,history]
llm=ChatOpenAI(model="gpt-4o-mini",api_key=OpenAI,max_tokens=15000)
ReAct_agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=agent_prompt
)