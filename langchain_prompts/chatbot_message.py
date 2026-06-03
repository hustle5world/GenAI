from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # ✅ much faster
    temperature=0.7,
    max_new_tokens=512,
)

chat_model = ChatHuggingFace(llm=llm)

messages = [SystemMessage(content="You are a helpful research assistant.")]

MAX_HISTORY_TURNS = 10  # keep last 10 Q&A pairs

def get_trimmed_messages(msgs, max_turns):
    """Keep system message + last N human/AI pairs."""
    system = [m for m in msgs if isinstance(m, SystemMessage)]
    history = [m for m in msgs if not isinstance(m, SystemMessage)]
    return system + history[-(max_turns * 2):]

while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))

    # ✅ Stream response token-by-token
    print("AI: ", end="", flush=True)
    full_response = ""

    try:
        trimmed = get_trimmed_messages(messages, MAX_HISTORY_TURNS)
        for chunk in chat_model.stream(trimmed):
            token = chunk.content
            print(token, end="", flush=True)
            full_response += token
        print()

    except Exception as e:
        print(f"\n[Error]: {e}")
        messages.pop()  # remove failed human message
        continue

    messages.append(AIMessage(content=full_response))

