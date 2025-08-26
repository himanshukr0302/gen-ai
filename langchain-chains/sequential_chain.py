import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("API KEY Not found")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# CODE

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'solo leveling'})
print(result)

# visualising the chain
chain.get_graph().print_ascii()