import streamlit as st
from graph import execute
from pymongo import MongoClient
from PIL import Image
import base64
from io import BytesIO
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot_database"]
collection = db["chat_history"]
from history import save_message
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
            if "content" in message:
                st.markdown(message["content"])

    prompt = st.text_input("Enter your question:")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("Submit"):
        if prompt:
            if uploaded_image:
                uploaded_image=Image.open(uploaded_image)
            res = execute(prompt,uploaded_image)
            print(res)
            st.session_state.messages.append({"role": "user", "content": prompt,"image":uploaded_image}) if uploaded_image else st.session_state.messages.append({"role": "user", "content": prompt})
            if type(res)==tuple:
                for img in res[1]:
                    st.session_state.messages.append({"role": "assistant", "image":img})
                st.session_state.messages.append({"role": "assistant", "content": res[0]})
                save_message("user", content=prompt)
                save_message("assistant", content=res[0], image=res[1][0])
            else:
                st.session_state.messages.append({"role": "assistant", "content": res})
                save_message("user", content=prompt)
                save_message("assistant", content=res)

        st.rerun()
