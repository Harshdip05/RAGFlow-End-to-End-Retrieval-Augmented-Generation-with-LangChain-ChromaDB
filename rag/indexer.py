# rag/indexer.py

from pathlib import Path

from .ingestion import (
    calculate_file_hash,
    load_pdf,
    split_documents
)

from .vectorstore import (
    add_documents_to_vectorstore
)

from .registry import (
    load_registry,
    save_registry
)


DOCS_FOLDER = Path("Docs")


def index_existing_documents():

    DOCS_FOLDER.mkdir(
        exist_ok=True
    )

    registry = load_registry()

    pdf_files = list(
        DOCS_FOLDER.glob("*.pdf")
    )

    results = []

    for pdf_file in pdf_files:

        # -----------------------------------------
        # Calculate unique hash
        # -----------------------------------------

        file_hash = calculate_file_hash(
            str(pdf_file)
        )


        # -----------------------------------------
        # Check if already indexed
        # -----------------------------------------

        if file_hash in registry:

            results.append({
                "filename": pdf_file.name,
                "status": "skipped",
                "reason": "Already indexed"
            })

            continue


        # -----------------------------------------
        # New document
        # -----------------------------------------

        document_id = file_hash


        # -----------------------------------------
        # Load PDF
        # -----------------------------------------

        docs = load_pdf(
            str(pdf_file),
            document_id,
            file_hash
        )


        # -----------------------------------------
        # Split PDF
        # -----------------------------------------

        chunks = split_documents(
            docs
        )


        # -----------------------------------------
        # Add to Chroma
        # -----------------------------------------

        add_documents_to_vectorstore(
            chunks
        )


        # -----------------------------------------
        # Save registry information
        # -----------------------------------------

        registry[file_hash] = {

            "filename": pdf_file.name,

            "document_id": document_id,

            "file_hash": file_hash,

            "pages": len(docs),

            "chunks": len(chunks)
        }


        save_registry(
            registry
        )


        results.append({

            "filename": pdf_file.name,

            "status": "indexed",

            "pages": len(docs),

            "chunks": len(chunks)
        })


    return results