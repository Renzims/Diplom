import streamlit as st
import replicate
import os
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
creds={"api_key":"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"}

llm = HuggingFaceEndpoint(
    repo_id="NousResearch/Llama-2-7b-chat-hf",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
    token=os.environ['HUGGINGFACEHUB_API_TOKEN']
)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
            """
            You are an AI assistant that provides helpful answers to user queries.
            """),
        ("user", "{question}\n"),
        ]
)
chain = prompt_template | llm
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Chat Bot", "AI_Photo" , "AI_Video"])

# Content for the Chat Bot page
if options == "Chat Bot":
    st.title("Echo Bot")

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    def generate_llama2_response(prompt_input):
        string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
            else:
                string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
        output = replicate.run(
            'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
            input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                   "temperature": 0.1, "top_p": 0.9, "max_length": 50, "repetition_penalty": 1})
        return output
    # React to user input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke({"question": prompt})
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

elif options == "AI_Photo":
    st.title("AI_Photo_Temp")
    st.write("""
        This is a simple Streamlit app that showcases how to build a chat bot and use 
        a sidebar for navigation. Streamlit is great for building quick prototypes and data apps!
        """)
elif options == "AI_Video":
    st.title("AI_Video_Temp")
    st.write("""
        This is a simple Streamlit app that showcases how to build a chat bot and use 
        a sidebar for navigation. Streamlit is great for building quick prototypes and data apps!
        """)
