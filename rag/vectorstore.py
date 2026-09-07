# rag/vectorstore.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


VECTOR_STORE_PATH = "./vector_store"


def get_embedding_model():

    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def get_vectorstore():

    embedding_model = get_embedding_model()

    vectorstore = Chroma(

        persist_directory=VECTOR_STORE_PATH,

        embedding_function=embedding_model
    )

    return vectorstore


def add_documents_to_vectorstore(chunks):

    vectorstore = get_vectorstore()

    vectorstore.add_documents(chunks)

    return vectorstore