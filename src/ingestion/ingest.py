"""
Ingestion pipeline: PDFs -> chunks -> embeddings -> persisted Chroma vector store.

Usage:
    python scripts/run_ingestion.py
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow `import config`

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config


def _load_pdfs_from_dir(directory: Path, source_type: str):
    """
    Loads each PDF individually (instead of one directory-wide call) so:
      - progress is visible per file, instead of looking frozen
      - a single slow/corrupt PDF can be skipped without killing the batch

    Uses PyMuPDF (fitz) rather than pypdf — pypdf can be extremely slow
    (minutes per page) on image-heavy, complex PDFs like TNAU/ICAR guides.
    """
    docs = []
    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        return docs

    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"  [{i}/{len(pdf_paths)}] Loading {pdf_path.name} ...", end=" ", flush=True)
        start = time.time()
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source_type"] = source_type
            docs.extend(file_docs)
            print(f"{len(file_docs)} pages in {time.time() - start:.1f}s")
        except Exception as e:
            print(f"SKIPPED (error: {e})")

    return docs


def load_all_documents():
    """Load PDFs from both the institutional-guide and Wikipedia directories,
    tagging each with a `source_type` metadata field so the retriever can
    later filter or weight institutional guidance over background reading."""
    docs = []

    if any(config.RAW_PDFS_DIR.glob("*.pdf")):
        print(f"Loading institutional guides from {config.RAW_PDFS_DIR}")
        docs.extend(_load_pdfs_from_dir(config.RAW_PDFS_DIR, "institutional_guide"))
    else:
        print(f"No PDFs found in {config.RAW_PDFS_DIR}, skipping.")

    if any(config.RAW_PDFS_WIKI_DIR.glob("*.pdf")):
        print(f"Loading background reading from {config.RAW_PDFS_WIKI_DIR}")
        docs.extend(_load_pdfs_from_dir(config.RAW_PDFS_WIKI_DIR, "background_reading"))
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