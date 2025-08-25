import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv() 
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found! Check your .env file.")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# code 

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me something about langchain'"),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)

