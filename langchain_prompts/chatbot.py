from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=256,
)

chat_model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
You are a helpful research assistant.

Question:
{question}

Provide a detailed answer.
""",
    input_variables=["question"]
)

st.title("Research Tool")

user_input = st.text_input("Enter your question")

if st.button("Research"):
    if user_input:
        formatted_prompt = prompt.format(question=user_input)

        response = chat_model.invoke(formatted_prompt)

        st.write(response.content)