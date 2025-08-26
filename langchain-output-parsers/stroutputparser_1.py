import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# For open source models - here we are using tinyllama
'''

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# loading model
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="conversational",
    huggingfacehub_api_token=hf_token
) # type: ignore

model = ChatHuggingFace(llm=llm)

'''

# for closed source like google's gemini
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Make sure GOOGLE_API_KEY is set in your .env file.")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# CODE -------------------------

# 1st prompt -> detailed report 
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
) #type: ignore

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
) #type: ignore


# using the output parser
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic': 'pokemon'})

print(result)