import base64
from io import BytesIO
from datetime import datetime
from pymongo import MongoClient
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