import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# CODE
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic'],
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})
print(result)

# visualisig the chain
chain.get_graph().print_ascii()
