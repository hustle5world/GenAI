from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=256,
)

chat_model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are a helpful research assistant."),
    HumanMessage(content="What is the difference between supervised and unsupervised learning?")
]

result=chat_model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)
