import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get the API key explicitly
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

# Pass the key directly to the model constructor
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', google_api_key=google_api_key)

result = model.invoke('What is the capital of India')

print(result.content)