from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='notes',
    glob='*.pdf',
    loader_cls=PyPDFLoader #type: ignore
)

# docs = loader.load()
# print(len(docs))
# print(docs[4].page_content)
# print(docs[5].metadata)

# lazy loader  - use this for loading large size documents in a limited memory
docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)