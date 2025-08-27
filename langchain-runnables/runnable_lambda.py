import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Key not found")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# FUN
word_count =  lambda x:len(x.split())

# CODE
parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'gdp of india'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)