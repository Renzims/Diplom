from langchain.agents import create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from Model_LLM_temp import model as llm_model  # ✅ Теперь это настоящий LangChain LLM!
from Model_Photo_temp import model as photo_model
from pymongo import MongoClient
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime

# Подключение к MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot_database"]
collection = db["chat_history"]


def save_message(role, content=None, image=None):
    """Сохраняет сообщения в MongoDB"""
    message = {"role": role, "content": content, "timestamp": datetime.utcnow()}
    if image:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        message["image"] = img_str
    collection.insert_one(message)


def fetch_chat_history(max_records=10):
    """Извлекает последние 10 пар сообщений из MongoDB"""
    messages = list(collection.find().sort("timestamp", -1).limit(max_records * 2))
    messages.reverse()
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages if 'content' in msg])
    return history


def generate_text_response(prompt: str):
    """Вызов LLM с раздельной передачей истории и запроса"""
    history = fetch_chat_history()
    response = llm_model(prompt)  # ✅ Теперь просто вызываем `llm_model()`

    save_message("user", content=prompt)
    save_message("assistant", content=response)
    return response


def analyze_image_response(prompt: str, image: Image.Image):
    """Анализ изображения с историей"""
    history = fetch_chat_history()
    response = llm_model(prompt, image=image)

    save_message("user", content=prompt, image=image)
    save_message("assistant", content=response)
    return response


def generate_photo(prompt: str):
    """Генерация изображения через Stable Diffusion"""
    image = photo_model(prompt, num_inference_steps=150).images[0]

    save_message("user", content=prompt)
    save_message("assistant", content="Generated image", image=image)
    return image


# ✅ Используем обычные Python-функции в `Tool`
tools = [
    Tool(name="TextGenerator", func=generate_text_response, description="Generates text responses."),
    Tool(name="PhotoGenerator", func=generate_photo, description="Generates an image based on the description."),
    Tool(name="ImageAnalyzer", func=analyze_image_response,
         description="Analyzes an image and provides a text response.")
]

# ✅ Новый `PromptTemplate`
agent_prompt = PromptTemplate(
    input_variables=["history", "input", "tools", "tool_names", "agent_scratchpad"],
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant. Your task is to generate appropriate responses based on the user's input and conversation history.

Chat History:
{history}

Available Tools:
{tools},{tool_names}

User Input:
{input}

Assistant Thought:
{agent_scratchpad}

Assistant Action:
"""
)

# ✅ Теперь `CustomLLM` полностью совместим с LangChain, можно передавать напрямую
agent_executor = create_react_agent(
    tools=tools,
    llm=llm_model,  # ✅ Теперь это LangChain LLM, передается без `RunnableLambda`
    prompt=agent_prompt,
)


def agent_route(prompt: str, image: Image.Image = None):
    """Функция маршрутизации запроса к агенту"""
    history = fetch_chat_history()

    if image:
        return agent_executor.invoke({
            "history": history,
            "input": f"Analyze this image with query: {prompt}",
            "tools": tools,
            "tool_names": [t.name for t in tools],
            "intermediate_steps": []
        })

    return agent_executor.invoke({
        "history": history,
        "input": prompt,
        "tools": tools,
        "tool_names": [t.name for t in tools],
        "intermediate_steps": []
    })
