# rag/ingestion.py

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def calculate_file_hash(file_path: str) -> str:
    """
    Generate a SHA-256 hash for the PDF.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while chunk := file.read(1024 * 1024):

            sha256.update(chunk)

    return sha256.hexdigest()


def load_pdf(
    file_path: str,
    document_id: str,
    file_hash: str
):
    """
    Load PDF and attach metadata.
    """

    loader = PyPDFLoader(file_path)

    docs = loader.load()

    source = Path(file_path).name

    for doc in docs:

        page_number = doc.metadata.get(
            "page",
            0
        )

        doc.metadata = {

            "source": source,

            "document_id": document_id,

            "file_hash": file_hash,

            "page": page_number + 1
        }

    return docs


def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    return chunks