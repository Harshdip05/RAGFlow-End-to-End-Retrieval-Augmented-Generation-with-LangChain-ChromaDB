# rag/qa.py

from langchain_openai import ChatOpenAI

from .vectorstore import get_vectorstore


def get_llm():

    return ChatOpenAI(
        model="gpt-5-nano",
        temperature=0
    )


def retrieve_documents(question, k=3):

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    return results


def generate_answer(question, documents):

    llm = get_llm()

    context = "\n\n".join(
        [
            doc.page_content
            for doc in documents
        ]
    )

    prompt = f"""
You are a helpful NISM study assistant.

Answer the user's question using ONLY the
provided document context.

If the answer cannot be found in the context,
say:

"I couldn't find sufficient information about
this question in the uploaded documents."

Do not invent information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content