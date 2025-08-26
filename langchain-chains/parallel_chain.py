import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("API KEY Not found")

model1 = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
model2 = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
model3 = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# CODE

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = '''
Pokémon Natures
Rather than just being a superficial personality, Natures actually affect the growth of a Pokémon. Each nature increases one of its stats by 10% and decreases one by 10% (by the time it reaches level 100). Five natures increase and decrease the same stat and therefore have no effect.

In most cases it is preferable to have a nature that decreases either Attack or Special Attack for Pokémon whose strengths are the opposite type of attack. Espeon, for example, favours Special moves, so it's best to use a nature that decreases its Attack since it won't be used.

Berries
A Pokémon's nature also determines the berries it likes and dislikes. Each type of berry is linked to one stat:

Attack - Spicy
Defense - Sour
Speed - Sweet
Sp. Attack - Dry
Sp. Defense - Bitter
The berry a Pokémon likes is the corresponding flavour of its raised stat, while the berry it dislikes is the flavour of its lowered stat.

For example, a Pokémon of Sassy nature will like Bitter berries (Special Defense is raised) and dislike Sweet berries (Speed is lowered).
'''

result = chain.invoke({'text':text})

print(result)

# visualising the chain
chain.get_graph().print_ascii()