# RAGForge

### End-to-End Retrieval-Augmented Generation with LangChain, OpenAI & ChromaDB

RAGForge is an end-to-end implementation of a **Retrieval-Augmented Generation (RAG)** pipeline built with Python, LangChain, OpenAI embeddings, and ChromaDB.

The project demonstrates how raw text and PDF documents can be transformed into embeddings, stored in a vector database, retrieved using semantic search, and used as contextual information for an LLM to generate grounded responses.

The implementation progresses from a basic in-memory RAG pipeline to persistent vector storage and incremental document ingestion.

---

## 🚀 What This Project Demonstrates

The project covers the core building blocks of a modern RAG system:

* Document ingestion
* Text preprocessing
* Document chunking
* Metadata management
* Text embeddings
* Vector databases
* Semantic similarity search
* Context retrieval
* LLM-powered question answering
* Persistent vector storage
* Reusing an existing vector database
* Incrementally adding new documents

---

## 🏗️ RAG Architecture

The overall pipeline follows this flow:

```text
                 ┌─────────────────┐
                 │  Source Data    │
                 │ Text / PDF      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Document Loader │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Text Splitter   │
                 │ Chunking        │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Embeddings    │
                 │ OpenAI          │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  ChromaDB       │
                 │ Vector Store    │
                 └────────┬────────┘
                          │
                    Semantic Search
                          │
                          ▼
                 ┌─────────────────┐
                 │ Relevant Chunks │
                 │    / Context    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │       LLM       │
                 │    OpenAI       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    Response     │
                 └─────────────────┘
```

---

## 📚 Project Structure

```text
RAGForge/
│
├── Docs/
│   ├── fabric-admin.pdf
│   └── fabric-onelake.pdf
│
├── Notes/
│   ├── RAG_Notes.png
│   └── readme.md
│
├── 1.0_RAG_With_Own_Text.ipynb
├── 2.0_RAG_PDF.ipynb
├── 3.0_RAG_Local_Persist.ipynb
├── 4.0_RAG_Add_Doc.ipynb
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

## 🧩 Implementation Stages

### 1. RAG with Custom Text

`1.0_RAG_With_Own_Text.ipynb`

The first implementation demonstrates the fundamental RAG workflow using custom text data.

Pipeline:

```text
Text
 ↓
Document
 ↓
Chunks
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Similarity Search
 ↓
LLM
 ↓
Answer
```

This notebook introduces the core concepts behind Retrieval-Augmented Generation.

---

### 2. RAG with PDF Documents

`2.0_RAG_PDF.ipynb`

The next stage extends the pipeline to real documents using PDF files.

The implementation uses:

* `PyPDFLoader` for PDF ingestion
* `RecursiveCharacterTextSplitter` for chunking
* `OpenAIEmbeddings` for vector representations
* `ChromaDB` for vector storage
* OpenAI LLM for response generation

Custom metadata is also attached to document chunks to preserve document-level information.

---

### 3. Persistent Vector Database

`3.0_RAG_Local_Persist.ipynb`

This stage introduces persistent vector storage.

Instead of creating a temporary vector store every time the notebook runs, embeddings are persisted locally.

```text
Documents
    ↓
Chunks
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Persistent Storage
```

The stored vector database can then be reopened and queried without recreating all embeddings.

This demonstrates an important step toward building a reusable RAG system.

---

### 4. Incrementally Add New Documents

`4.0_RAG_Add_Doc.ipynb`

The final stage demonstrates how a new document can be added to an existing vector database.

Instead of rebuilding the entire vector store:

```text
Existing Vector DB
        +
   New Document
        ↓
   New Chunks
        ↓
   New Embeddings
        ↓
Updated Vector DB
```

This approach is useful when a knowledge base continuously receives new documents.

---

## 🛠️ Tech Stack

| Technology       | Purpose                                   |
| ---------------- | ----------------------------------------- |
| Python           | Core programming language                 |
| LangChain        | RAG orchestration and document processing |
| OpenAI           | LLM and embedding generation              |
| ChromaDB         | Vector database                           |
| PyPDF            | PDF document processing                   |
| Jupyter Notebook | Experimentation and implementation        |

---

## 🔑 Core RAG Concepts

### Documents

Raw information supplied to the RAG pipeline, such as text or PDF files.

### Chunking

Large documents are divided into smaller pieces so that relevant sections can be efficiently retrieved.

Example:

```text
Large Document
      ↓
 ┌──────────┐
 │ Chunk 1  │
 ├──────────┤
 │ Chunk 2  │
 ├──────────┤
 │ Chunk 3  │
 └──────────┘
```

### Embeddings

Each chunk is converted into a numerical vector representing its semantic meaning.

```text
"Microsoft Fabric provides..."
              ↓
       Embedding Model
              ↓
    [0.12, -0.45, 0.78, ...]
```

### Vector Database

The generated embeddings are stored in ChromaDB so that semantically similar content can be retrieved efficiently.

### Semantic Search

Instead of matching exact keywords, the system searches for chunks that are semantically related to the user's query.

### Retrieval

The most relevant chunks are selected from the vector database.

### Generation

The retrieved context is provided to the LLM so that it can generate an answer based on the available information.

---

## 🔄 Complete RAG Workflow

```text
                INGESTION
                    │
                    ▼
          ┌─────────────────┐
          │ Text / PDF Data │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Document Loader │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │     Chunking    │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   Embeddings    │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │    ChromaDB     │
          └────────┬────────┘
                   │
                   │
              RETRIEVAL
                   │
                   ▼
          ┌─────────────────┐
          │ User Question   │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Semantic Search │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Relevant Chunks │
          └────────┬────────┘
                   │
                   ▼
              GENERATION
                   │
                   ▼
          ┌─────────────────┐
          │      LLM        │
          └────────┬────────┘
                   │
                   ▼
             Final Answer
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ragforge.git

cd ragforge
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file based on `.env.example`.

```env
OPENAI_API_KEY=your_openai_api_key
```

Never commit your actual API key to GitHub.

---

## ▶️ Running the Project

The project is organized as a sequence of Jupyter notebooks.

Start with:

```text
1.0_RAG_With_Own_Text.ipynb
```

Then progress through:

```text
2.0_RAG_PDF.ipynb
        ↓
3.0_RAG_Local_Persist.ipynb
        ↓
4.0_RAG_Add_Doc.ipynb
```

Each notebook builds on the concepts introduced previously.

---

## 📖 Learning Path

If you are new to RAG, follow this order:

### Step 1 — Understand the fundamentals

Learn:

* What is RAG?
* Why RAG is needed
* Embeddings
* Vector databases
* Similarity search
* Retrieval vs generation

### Step 2 — Build basic RAG

Run:

```text
1.0_RAG_With_Own_Text.ipynb
```

### Step 3 — Work with real documents

Run:

```text
2.0_RAG_PDF.ipynb
```

### Step 4 — Introduce persistence

Run:

```text
3.0_RAG_Local_Persist.ipynb
```

### Step 5 — Build an updatable knowledge base

Run:

```text
4.0_RAG_Add_Doc.ipynb
```

---

## 🎯 Key Learning Outcomes

After completing this project, you should understand:

* How a RAG pipeline works end-to-end
* Why documents need to be chunked
* How embeddings represent semantic meaning
* How vector databases store embeddings
* How similarity search retrieves relevant information
* How retrieved context is passed to an LLM
* How persistent vector databases work
* How new documents can be added to an existing knowledge base
* The difference between ingestion, retrieval, and generation

---

## 🚀 Future Improvements

Possible extensions for turning this implementation into a more production-oriented RAG system include:

* Hybrid search
* Metadata filtering
* Reranking
* Query rewriting
* Conversational memory
* Source citations
* Retrieval evaluation
* RAG evaluation metrics
* FastAPI backend
* Web-based chat interface
* Authentication
* Observability and tracing
* Automated document ingestion
* Background indexing pipelines

---

## 💡 Example Use Cases

The same architecture can be adapted for:

* Internal company knowledge bases
* Technical documentation assistants
* PDF question-answering systems
* Data engineering documentation
* Research assistants
* Customer support knowledge bases
* Enterprise document search
* Policy and compliance assistants

---

## 📌 Why RAG?

Large Language Models have strong general knowledge, but they may not know about an organization's private or recently updated information.

RAG addresses this by retrieving relevant information from an external knowledge base and providing it to the LLM as context.

```text
User Question
      ↓
Retrieve Relevant Knowledge
      ↓
Provide Knowledge as Context
      ↓
LLM Generates Answer
```

This allows applications to work with information that is outside the model's original training data.

---

## 👨‍💻 Author

**Harshdip Nandre**

If you found this project useful, consider giving the repository a ⭐.

---

## 📄 License

This project is intended for educational and demonstration purposes.
