import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st 

load_dotenv() 

# Verify that the key is being loaded
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Make sure GOOGLE_API_KEY is set in your .env file.")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# adding ui elements
st.header("Reseach Tool")

user_input = st.text_input("Enter your prompt here")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content)

