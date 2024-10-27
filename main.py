import streamlit as st
from Model_LLM import model as llm_model
#from Model_Photo import model
from Model_Photo import model
import sys
import random
import torch
from PIL import Image
# Создаем боковую панель для навигации
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Chat Bot", "AI_Photo", "AI_Video"])

# Контент для страницы Chat Bot
if options == "Chat Bot":
    st.title("Echo Bot")

    # Инициализация истории чата
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Отображение истории сообщений
    for message in st.session_state.messages:
        if message["role"] == "user" and "image" in message:
            # Отображение загруженного изображения
            with st.chat_message(message["role"]):
                st.image(message["image"], caption="User uploaded an image.")
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    prompt = st.text_input("Enter your question:")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Обработка ввода пользователя
    if st.button("Submit"):
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Проверка наличия изображения
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.session_state.messages.append({"role": "user", "image": image})
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Отправка запроса с изображением
                        response = llm_model._call(prompt,image)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Отправка запроса без изображения
                        response = llm_model._call(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})



elif options == "AI_Photo":
    st.title("Генерация изображений с помощью Hugging Face Stable Diffusion")
    prompt = st.text_input("Введите текстовое описание:")
    if st.button("Сгенерировать изображение"):
        if prompt:
            with st.spinner("Генерация изображения..."):
                '''
                seed = random.randint(0, sys.maxsize)
                num_inference_steps = 150

                pipeline_params = {
                    "prompt": prompt,
                    "output_type": "pil",
                    "generator": torch.Generator("cuda").manual_seed(seed),
                    "num_inference_steps": num_inference_steps,
                }
                '''
                #images = pipe(**pipeline_params).images
                image = model(prompt,num_inference_steps=150).images[0]
                #image = images[0]
            st.image(image, caption="Сгенерированное изображение", use_column_width=True)
        else:
            st.error("Пожалуйста, введите текстовое описание.")

elif options == "AI_Video":
    st.title("AI_Video_Temp")
    st.write("""
        This is a simple Streamlit app that showcases how to build a chat bot and use 
        a sidebar for navigation. Streamlit is great for building quick prototypes and data apps!
    """)