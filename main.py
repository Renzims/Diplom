import streamlit as st
from Model_LLM import chain
from Model_Photo import model
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
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Реакция на ввод пользователя
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke({"question": prompt})  # Запуск цепочки с моделью Hugging Face
                full_response = "".join(response) if isinstance(response, list) else response

                # Отображение ответа
                st.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)


elif options == "AI_Photo":
    st.title("Генерация изображений с помощью Hugging Face Stable Diffusion")
    prompt = st.text_input("Введите текстовое описание:")
    if st.button("Сгенерировать изображение"):
        if prompt:
            with st.spinner("Генерация изображения..."):
                # Генерируем изображение
                image = model(prompt,num_inference_steps=350).images[0]

            st.image(image, caption="Сгенерированное изображение", use_column_width=True)
        else:
            st.error("Пожалуйста, введите текстовое описание.")

elif options == "AI_Video":
    st.title("AI_Video_Temp")
    st.write("""
        This is a simple Streamlit app that showcases how to build a chat bot and use 
        a sidebar for navigation. Streamlit is great for building quick prototypes and data apps!
    """)