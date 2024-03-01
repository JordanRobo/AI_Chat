import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__e777165acb31400eb429d680efabe41a"

# load the document and split it into chunks
loader = DirectoryLoader("app/afereon", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# save it into Chroma
db = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_function, 
    persist_directory="app/chroma_db"
    )
