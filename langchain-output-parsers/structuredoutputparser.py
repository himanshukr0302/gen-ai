import os 
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Make sure GOOGLE_API_KEY is set in your .env file.")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# CODE
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

'''
prompt = template.invoke({'topic':'black hole'})
result = model.invoke(prompt)
final_result = parser.parse(result.content) #type:ignore
print(final_result)
'''

# same code above using chain
chain = template | model | parser
result = chain.invoke({'topic':'black hole'})
print(result)