"""
Ingestion pipeline: PDFs -> chunks -> embeddings -> persisted Chroma vector store.

Usage:
    python scripts/run_ingestion.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow `import config`

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config


def load_all_documents():
    """Load PDFs from both the institutional-guide and Wikipedia directories,
    tagging each with a `source_type` metadata field so the retriever can
    later filter or weight institutional guidance over background reading."""
    docs = []

    if any(config.RAW_PDFS_DIR.glob("*.pdf")):
        loader = PyPDFDirectoryLoader(str(config.RAW_PDFS_DIR))
        institutional_docs = loader.load()
        for d in institutional_docs:
            d.metadata["source_type"] = "institutional_guide"
        docs.extend(institutional_docs)
        print(f"Loaded {len(institutional_docs)} pages from {config.RAW_PDFS_DIR}")
    else:
        print(f"No PDFs found in {config.RAW_PDFS_DIR}, skipping.")

    if any(config.RAW_PDFS_WIKI_DIR.glob("*.pdf")):
        loader = PyPDFDirectoryLoader(str(config.RAW_PDFS_WIKI_DIR))
        wiki_docs = loader.load()
        for d in wiki_docs:
            d.metadata["source_type"] = "background_reading"
        docs.extend(wiki_docs)
        print(f"Loaded {len(wiki_docs)} pages from {config.RAW_PDFS_WIKI_DIR}")
    else:
        print(f"No PDFs found in {config.RAW_PDFS_WIKI_DIR}, skipping.")

    if not docs:
        raise FileNotFoundError(
            "No PDFs found in data/raw_pdfs or data/raw_pdfs_wikipedia. "
            "Add source PDFs before running ingestion."
        )

    return docs


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(config.CHROMA_PERSIST_DIR),
        collection_name=config.COLLECTION_NAME,
    )
    print(f"Persisted vector store to {config.CHROMA_PERSIST_DIR} "
          f"(collection='{config.COLLECTION_NAME}')")
    return vectorstore


def run():
    docs = load_all_documents()
    build_vectorstore(docs)
    print("Ingestion complete.")


if __name__ == "__main__":
    run()
