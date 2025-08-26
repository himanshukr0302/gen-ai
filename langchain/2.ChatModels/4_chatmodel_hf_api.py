import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

# Load env vars
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env")

# Use HuggingFaceEndpoint with conversational task
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="conversational",
    huggingfacehub_api_token=hf_token
) # type: ignore

# Wrap inside ChatHuggingFace (correct class for chat models)
chat_model = ChatHuggingFace(llm=llm)

# Send a chat-style message
result = chat_model.invoke([HumanMessage(content="What is the capital of India?")])
print(result.content)
