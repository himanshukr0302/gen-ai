import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Key not found")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# FUN
word_count =  lambda x:len(x.split())

# CODE
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, model, parser)), # type: ignore
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic':'black hole'})
print(result)
