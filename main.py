import streamlit as st
from Model_LLM import model as llm_model
#from Model_Photo import model
from Model_Photo import model as photo_model
from pymongo import MongoClient
from datetime import datetime
from PIL import Image
from io import BytesIO
import base64

client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot_database"]
collection = db["chat_history"]

def save_message(role, content=None, image=None):
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    }
    if image:
        # Конвертируем изображение в формат Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        message["image"] = img_str  # Сохраняем строку Base64

    collection.insert_one(message)

# Функция для загрузки всей истории чата из MongoDB
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

# Контент для страницы Chat Bot
if options == "Chat Bot":
    st.title("Echo Bot")

    # Инициализация истории чата
    if "messages" not in st.session_state:
        # Загружаем историю из базы данных при запуске
        st.session_state.messages = (load_chat_history() or
                                     [{"role": "assistant", "content": "How may I assist you today?"}])
    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    # Отображение истории сообщений
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user" and "image" in message:
                st.markdown(message["content"] if message["content"] else "User sent an image")
                st.image(message["image"], caption="User uploaded an image.")
            elif message["role"] == "assistant" and "image" in message:
                st.markdown(message["content"] if message["content"] else "assistant generate an image")
                st.image(message["image"], caption="assistant generate an image")
            else:
                # Отображаем только текст
                st.markdown(message["content"])
    prompt = st.text_input("Enter your question:")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Обработка ввода пользователя
    if st.button("Submit"):
        if prompt:
            if prompt.startswith("Generate_photo:"):
                # Извлекаем текст после команды
                user_response = prompt[len("Generate_photo:"):].strip()
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("Generating photo..."):
                        # Вызов модели для генерации фото
                        generated_image = photo_model(user_response,num_inference_steps=150).images[0]
                        response = "Here is your generated image."
                        st.session_state.messages.append({"role": "assistant", "content": response,"image": generated_image})
                        save_message("user", content=prompt)
                        save_message("assistant", content=response,image=generated_image)
            else:
                if uploaded_image is not None:
                    image = Image.open(uploaded_image)
                    st.session_state.messages.append({"role": "user", "content": prompt, "image": image})
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                        # Отправка запроса с изображением
                            response = llm_model._call(prompt,image)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            save_message("user", content=prompt, image=image)
                            save_message("assistant", content=response)

                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                        # Отправка запроса без изображения
                            response = llm_model._call(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            save_message("user", content=prompt)
                            save_message("assistant", content=response)
        st.session_state.prompt = ""
        st.session_state.uploaded_image = None
        st.rerun()