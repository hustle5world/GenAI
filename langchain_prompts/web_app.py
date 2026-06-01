from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=256,
)

chat_model = ChatHuggingFace(llm=llm)

st.title("Chat with Hugging Face")

prompt = st.text_input("Enter Prompt")

if st.button("Generate"):
    if prompt:
        response = chat_model.invoke(prompt)

        st.write("Response:")
        st.write(response.content)
    else:
        st.warning("Please enter a prompt.")