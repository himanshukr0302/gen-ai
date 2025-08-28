from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# for text from another document
loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

# for text document within the same file
text = """
Individual Values (IVs for short) are key to determining your Pokémon's stats. IVs are randomly assigned when you encounter any wild Pokémon (or receive it as a gift or egg). You cannot change a Pokémon's IVs once caught and there is no way to know the exact IVs in-game.

IVs consist of six numbers corresponding to HP, Attack, Defense, Special Attack, Special Defense and Speed. Each range from 0-31 and the higher the number, the better your Pokémon's stat will be.

IVs also determine the type and strength of the move Hidden Power. In Pokémon Platinum, HeartGold and SoulSilver, you can find out what type Hidden Power would be for any Pokémon by showing it to a man in the game corner Prize Exchange building (Veilstone City for Platinum, Celadon City for HG/SS). 
"""


splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0,
    separator=''
)

# for text document within the same file
splitted_sentences = splitter.split_text(text)
print(splitted_sentences)

print('\n')

# for text document from the another file
docs_splitted_text = splitter.split_documents(docs)
print(docs_splitted_text[3].page_content)
