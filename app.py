import os
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# ============================================================
# CONFIGURATION
# ============================================================

# Load variables from .env when running locally.
#
# Local:
#     .env
#     OPENAI_API_KEY=your_key
#
# Streamlit Cloud:
#     st.secrets["OPENAI_API_KEY"]
#
# This allows the same application to work both locally
# and after deployment.
load_dotenv()


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NISM RAG Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

DOCS_DIR = Path("Docs")
VECTOR_STORE_DIR = Path("vector_store")

DOCS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5-nano"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

DEFAULT_K = 3


# ============================================================
# API KEY CONFIGURATION
# ============================================================

# First try to get the API key from Streamlit Secrets.
#
# This is used when the application is deployed on
# Streamlit Community Cloud.
#
# Example Streamlit Secret:
#
# OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"
#
# If Streamlit Secrets are not available, the application
# will continue using the value loaded from .env.

try:

    if "OPENAI_API_KEY" in st.secrets:

        os.environ["OPENAI_API_KEY"] = (
            st.secrets["OPENAI_API_KEY"]
        )

except Exception:
    # st.secrets may not be configured when running locally.
    # In that case, continue using the .env value.
    pass


# ============================================================
# API KEY CHECK
# ============================================================

if not os.getenv("OPENAI_API_KEY"):

    st.error(
        "OPENAI_API_KEY is not configured.\n\n"
        "For local development, add it to your .env file.\n"
        "For Streamlit Cloud, add it to App Settings → Secrets."
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "retrieval_k" not in st.session_state:

    st.session_state.retrieval_k = DEFAULT_K


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #888;
        margin-bottom: 25px;
    }

    .stat-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        margin-bottom: 10px;
    }

    .stat-number {
        font-size: 28px;
        font-weight: 700;
    }

    .stat-label {
        font-size: 14px;
        color: #888;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


@st.cache_resource
def get_embeddings():

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )


@st.cache_resource
def get_llm():

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
    )


def get_vectorstore():

    return Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=get_embeddings(),
    )


def get_file_hash(file_path: Path):

    """
    Create a unique SHA-256 hash for a PDF.

    This helps us detect duplicate documents.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def load_and_split_pdf(
    file_path: Path,
    document_id: str,
):

    """
    Load PDF and split it into chunks.
    """

    loader = PyPDFLoader(
        str(file_path)
    )

    documents = loader.load()

    # --------------------------------------------------------
    # Add metadata BEFORE splitting
    # --------------------------------------------------------

    for document in documents:

        document.metadata["source"] = (
            file_path.name
        )

        document.metadata["document_id"] = (
            document_id
        )

        document.metadata["page"] = (
            document.metadata.get("page", 0) + 1
        )

    # --------------------------------------------------------
    # Split documents into chunks
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


def get_indexed_document_ids():

    """
    Get document IDs already stored in Chroma.
    """

    try:

        vectorstore = get_vectorstore()

        data = vectorstore.get(
            include=["metadatas"]
        )

        metadatas = data.get(
            "metadatas",
            []
        )

        document_ids = set()

        for metadata in metadatas:

            if metadata:

                document_id = metadata.get(
                    "document_id"
                )

                if document_id:

                    document_ids.add(
                        document_id
                    )

        return document_ids

    except Exception:

        return set()


def add_pdf_to_vectorstore(
    file_path: Path,
):

    """
    Add one PDF to Chroma.
    """

    # --------------------------------------------------------
    # Generate SHA-256 hash
    # --------------------------------------------------------

    file_hash = get_file_hash(
        file_path
    )

    # --------------------------------------------------------
    # Check whether document already exists
    # --------------------------------------------------------

    indexed_ids = get_indexed_document_ids()

    if file_hash in indexed_ids:

        return {
            "status": "exists",
            "chunks": 0,
        }

    # --------------------------------------------------------
    # Load and split PDF
    # --------------------------------------------------------

    chunks = load_and_split_pdf(
        file_path,
        file_hash,
    )

    if not chunks:

        return {
            "status": "empty",
            "chunks": 0,
        }

    # --------------------------------------------------------
    # Add chunks to ChromaDB
    # --------------------------------------------------------

    vectorstore = get_vectorstore()

    vectorstore.add_documents(
        documents=chunks
    )

    return {
        "status": "added",
        "chunks": len(chunks),
    }


def retrieve_documents(
    question: str,
    k: int,
):

    """
    Retrieve the most relevant document chunks
    from ChromaDB.
    """

    vectorstore = get_vectorstore()

    documents = vectorstore.similarity_search(
        question,
        k=k,
    )

    return documents


def generate_answer(
    question: str,
    documents,
):

    """
    Generate an answer using only the retrieved
    document context.
    """

    if not documents:

        return (
            "I couldn't find sufficient information "
            "about this question in the uploaded documents."
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        page = document.metadata.get(
            "page",
            "Unknown",
        )

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{document.page_content}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Build RAG prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful NISM study assistant.

Answer the user's question using ONLY the
provided document context.

Rules:
1. Do not invent information.
2. If the answer is not present in the context,
   clearly say that you could not find sufficient
   information in the uploaded documents.
3. Give a clear and concise answer.
4. Use the source information when useful.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # --------------------------------------------------------
    # Generate answer using OpenAI
    # --------------------------------------------------------

    response = get_llm().invoke(
        prompt
    )

    return response.content


def get_document_count():

    try:

        files = list(
            DOCS_DIR.glob("*.pdf")
        )

        return len(files)

    except Exception:

        return 0


def get_chunk_count():

    try:

        vectorstore = get_vectorstore()

        return vectorstore._collection.count()

    except Exception:

        return 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 📚 NISM RAG")

    st.caption(
        "AI-powered NISM document assistant"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "💬 Chat",
            "📤 Add PDF",
            "📚 Documents",
            "⚙️ Settings",
        ],
    )

    st.divider()

    st.markdown("### Knowledge Base")

    st.metric(
        "PDF Documents",
        get_document_count(),
    )

    st.metric(
        "Indexed Chunks",
        get_chunk_count(),
    )

    st.divider()

    st.caption(
        "LangChain • ChromaDB • OpenAI"
    )


# ============================================================
# CHAT PAGE
# ============================================================

if page == "💬 Chat":

    st.markdown(
        '<div class="main-title">'
        '📚 NISM AI Assistant'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Ask questions from your NISM documents'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {get_document_count()}
                </div>
                <div class="stat-label">
                    PDF Documents
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {get_chunk_count()}
                </div>
                <div class="stat-label">
                    Indexed Chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {EMBEDDING_MODEL}
                </div>
                <div class="stat-label">
                    Embedding Model
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # --------------------------------------------------------
    # Existing chat messages
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # User question
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask something about your NISM documents..."
    )

    if question:

        # Store user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching your documents..."
            ):

                try:

                    documents = retrieve_documents(
                        question,
                        st.session_state.retrieval_k,
                    )

                    answer = generate_answer(
                        question,
                        documents,
                    )

                    st.markdown(answer)

                    # ----------------------------------------
                    # Sources
                    # ----------------------------------------

                    if documents:

                        st.markdown(
                            "### 📚 Sources"
                        )

                        for index, document in enumerate(
                            documents,
                            start=1,
                        ):

                            source = document.metadata.get(
                                "source",
                                "Unknown",
                            )

                            page_number = document.metadata.get(
                                "page",
                                "Unknown",
                            )

                            with st.expander(
                                f"📄 {source} — Page {page_number}"
                            ):

                                st.write(
                                    document.page_content
                                )

                except Exception as error:

                    answer = (
                        "Something went wrong while "
                        "searching the knowledge base."
                    )

                    st.error(
                        f"{answer}\n\n{error}"
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


# ============================================================
# ADD PDF PAGE
# ============================================================

elif page == "📤 Add PDF":

    st.title("📤 Add PDF")

    st.caption(
        "Upload a new PDF and add it to the "
        "existing Chroma knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        st.subheader(
            "Selected documents"
        )

        for uploaded_file in uploaded_files:

            st.write(
                f"📄 **{uploaded_file.name}** "
                f"— "
                f"{uploaded_file.size / 1024:.1f} KB"
            )

        st.divider()

        if st.button(
            "🚀 Add to Knowledge Base",
            type="primary",
            use_container_width=True,
        ):

            progress = st.progress(0)

            status = st.empty()

            total = len(
                uploaded_files
            )

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                status.info(
                    f"Processing {uploaded_file.name}..."
                )

                # --------------------------------------------
                # Save uploaded PDF
                # --------------------------------------------

                file_path = (
                    DOCS_DIR /
                    uploaded_file.name
                )

                with open(
                    file_path,
                    "wb",
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                # --------------------------------------------
                # Add to Chroma
                # --------------------------------------------

                try:

                    result = add_pdf_to_vectorstore(
                        file_path
                    )

                    if result["status"] == "added":

                        st.success(
                            f"✅ {uploaded_file.name} "
                            f"added successfully. "
                            f"{result['chunks']} chunks indexed."
                        )

                    elif result["status"] == "exists":

                        st.info(
                            f"ℹ️ {uploaded_file.name} "
                            f"is already indexed."
                        )

                    elif result["status"] == "empty":

                        st.warning(
                            f"⚠️ {uploaded_file.name} "
                            f"does not contain readable text."
                        )

                except Exception as error:

                    st.error(
                        f"❌ Failed to process "
                        f"{uploaded_file.name}: {error}"
                    )

                progress.progress(
                    (index + 1) / total
                )

            status.success(
                "🎉 Processing completed!"
            )

            st.rerun()


# ============================================================
# DOCUMENTS PAGE
# ============================================================

elif page == "📚 Documents":

    st.title("📚 Document Library")

    st.caption(
        "PDF files currently stored in your Docs folder."
    )

    pdf_files = list(
        DOCS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        st.info(
            "No PDF documents found."
        )

    else:

        st.write(
            f"### {len(pdf_files)} document(s)"
        )

        for pdf_file in pdf_files:

            with st.container(
                border=True
            ):

                col1, col2 = st.columns(
                    [5, 2]
                )

                with col1:

                    st.markdown(
                        f"### 📄 {pdf_file.name}"
                    )

                    st.caption(
                        f"Size: "
                        f"{pdf_file.stat().st_size / 1024:.1f} KB"
                    )

                with col2:

                    st.write(
                        "Indexed"
                    )

                    st.write(
                        "PDF document"
                    )


# ============================================================
# SETTINGS PAGE
# ============================================================

elif page == "⚙️ Settings":

    st.title("⚙️ Settings")

    st.subheader(
        "Retrieval Settings"
    )

    st.session_state.retrieval_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=st.session_state.retrieval_k,
    )

    st.caption(
        "Higher values provide more context to the LLM "
        "but may increase prompt size."
    )

    st.divider()

    st.subheader(
        "Embedding Model"
    )

    st.code(
        EMBEDDING_MODEL
    )

    st.subheader(
        "Language Model"
    )

    st.code(
        LLM_MODEL
    )

    st.divider()

    st.subheader(
        "Chunking Configuration"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Chunk Size",
            CHUNK_SIZE,
        )

    with col2:

        st.metric(
            "Chunk Overlap",
            CHUNK_OVERLAP,
        )

    st.info(
        "Changing the embedding model, chunk size, "
        "or chunk overlap after documents have been "
        "indexed may require rebuilding the vector store."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NISM RAG Assistant • "
    "Retrieval-Augmented Generation with "
    "LangChain + ChromaDB + OpenAI"
)
