import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

# Just to remove a little warning sign - although not required
os.environ['USER_AGENT'] = "Your-Custom-App-Name/1.0"

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Key not found")

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

url = "https://glorioushinduism.com/2023/10/19/mahabharata-family-tree/"
loader = WebBaseLoader(url)
docs = loader.load()

# Using Model
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=['question','text']
)

chain = prompt | model | parser

result = chain.invoke({'question':"Who were the kings in generation 5?",'text':docs[0].page_content}) # type: ignore 

print(result)

