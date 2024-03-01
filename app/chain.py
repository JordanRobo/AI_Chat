import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__e777165acb31400eb429d680efabe41a"

# create the open-source embedding function
embedding_function = OllamaEmbeddings(model="nomic-embed-text")

# load from disk
vectore_store = Chroma(persist_directory="app/chroma_db", embedding_function=embedding_function)
retriever = vectore_store.as_retriever()

template = """Answer the following question based only on the provided context:
{context}

Question: {question}
"""

# Create AI chat prompt
prompt = ChatPromptTemplate.from_template(template)

# Define the AI model
ollama_llm = "hub/bagellama/d&d-dungeon-master-assistant:latest"
llm = ChatOllama(model=ollama_llm)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)