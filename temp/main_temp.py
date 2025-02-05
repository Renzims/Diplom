import streamlit as st
from agent_router_temp import agent_route
from pymongo import MongoClient
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
# После того как все будет работать упростить этот файл
# Подключение к MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot_database"]
collection = db["chat_history"]


# Функция загрузки истории чата
def load_chat_history():
    messages = []
    for doc in collection.find():
        message = {"role": doc["role"], "content": doc["content"]}
        if "image" in doc:
            image_data = base64.b64decode(doc["image"])
            image = Image.open(BytesIO(image_data))
            message["image"] = image
        messages.append(message)
    return messages


# Streamlit UI
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Chat Bot", "AI_Photo"])

if options == "Chat Bot":
    st.title("Smart AI Chat Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history() or [
            {"role": "assistant", "content": "How may I assist you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "image" in message:
                st.image(message["image"])
            st.markdown(message["content"])

    prompt = st.text_input("Enter your question:")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("Submit"):
        if prompt:
            if uploaded_image:
                image = Image.open(uploaded_image)
                response = agent_route(prompt, image)
                st.session_state.messages.append({"role": "user", "content": prompt, "image": image})
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = agent_route(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response})

        st.rerun()
