"""
LCEL RAG chain: retriever -> prompt -> Groq LLM -> string output.

This mirrors the architecture in the implementation plan, with the LLM
swapped for Groq (`langchain-groq`), which implements the same
`BaseChatModel` interface so the rest of the LCEL chain is unchanged.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

import config

PROMPT_TEMPLATE = """
You are an expert Agricultural Advisory AI assisting farmers.
Use the following retrieved institutional guidelines and research contexts to
answer the farmer's question accurately. If the context does not contain the
answer, say so plainly rather than guessing.

Retrieved Context:
{context}

Farmer Question: {question}

Provide clear, step-by-step advice covering:
1. Direct Answer & Explanation
2. Recommended Action / Fertilizer / Treatment Schedule
3. Safety & Preventive Precautions
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(config.CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    )


def build_rag_chain(metadata_filter: dict | None = None):
    """Build the LCEL RAG chain.

    metadata_filter: optional Chroma `where` filter, e.g.
        {"source_type": "institutional_guide"}
    to prefer official guidance over background reading for a given query.
    """
    vectorstore = load_vectorstore()

    search_kwargs = {"k": config.RETRIEVER_K}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = vectorstore.as_retriever(
        search_type=config.SEARCH_TYPE,
        search_kwargs=search_kwargs,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    llm = ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE,
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


if __name__ == "__main__":
    chain = build_rag_chain()
    question = "My tomato leaves have brown concentric rings, what should I do?"
    print(chain.invoke(question))
