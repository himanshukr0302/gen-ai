import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Key not found")
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

parser = StrOutputParser()

# CODE
loader = TextLoader('ivs.txt', encoding='utf-8')
docs = loader.load()


print(len(docs))
print(type(docs[0]))

print(docs[0].metadata)
print(docs[0].page_content)

# Using the model
prompt = PromptTemplate(
    template="Write a summary of the following text, \n {text}",
    input_variables=['text']
)

chain = prompt | model | parser
result = chain.invoke({'text':docs[0].page_content})

print(result)