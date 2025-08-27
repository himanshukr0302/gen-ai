import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Key not found")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# CODE
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}, and do not give me any options',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Generata a Linkedin post about {topic}, and do not give me any options',
    input_variables=['topic']
)

parallel_chain = RunnableParallel(
    {
       'tweet': RunnableSequence(prompt1,model,parser),
       'linkedin': RunnableSequence(prompt2,model,parser)
    }
)

result = parallel_chain.invoke({'topic':'AI'})

print("Your Tweet: ",result['tweet'])
print('\n')
print("Your Linkedin post: ",result['linkedin'])
