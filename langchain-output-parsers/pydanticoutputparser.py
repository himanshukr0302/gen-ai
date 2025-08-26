import os 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Make sure GOOGLE_API_KEY is set in your .env file.")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# CODE
class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

'''
prompt = template.invoke({'place':'indian'})
result = model.invoke(prompt)
final_result = parser.parse(result.content) #type:ignore
print(final_result)
'''

# writting the same code above using chain
chain = template | model | parser
result = chain.invoke({'place':'bihari'})
print(result)