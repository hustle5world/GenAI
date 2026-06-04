from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    temperature=1
    # max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write a 5-line summary of the following text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain1 = prompt1 | model | parser
chain2 = prompt2 | model | parser

report = chain1.invoke({"topic": "black hole"})

print("REPORT:")
print(report)

summary = chain2.invoke({"text": report})

print("\nSUMMARY:")
print(summary)