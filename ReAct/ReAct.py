from langchain import hub
from langchain.agents import create_react_agent
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from Model_Photo_temp import model as photo_model
from Model_LLM_temp import model, template
from PIL import Image
import json


OpenAI="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"

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


agent_prompt = hub.pull("hwchase17/react")
tools=[generate_photo,image_analysis]
llm=ChatOpenAI(model="gpt-4o-mini",api_key=OpenAI)
ReAct_agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=agent_prompt
)
