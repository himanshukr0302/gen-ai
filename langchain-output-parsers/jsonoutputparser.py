import os 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Make sure GOOGLE_API_KEY is set in your .env file.")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# prompts 

parser = JsonOutputParser()

template = PromptTemplate(
    template='Guess a random pokemon with it\' type: \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions}
) # type:ignore 

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content) # type: ignore

# print(final_result)
# print(type(final_result))

# using chains for the same above task
chain = template | model | parser

result = chain.invoke({})

print(result)