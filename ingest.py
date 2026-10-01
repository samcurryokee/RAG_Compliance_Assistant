"""Load PDFs -> chunk -> embed with Gemini -> store in pgvector.

Usage:  python3 ingest.py
Each run WIPES and rebuilds the collection, so it is safe to re-run.
Batched and paced to stay under the Gemini free-tier limit (100 requests/minute).
"""
import glob
import os
import sys
import time

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

from chain import COLLECTION_NAME, EMBEDDING_MODEL, get_engine  # noqa: E402

DOCS_DIR = "data/documents"
BATCH_SIZE = 80
PAUSE_SECONDS = 65


def load_pdfs():
    """One Document per PDF page. metadata: source (path) and page (0-indexed)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf"))):
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": path, "page": i}))
    print(f"Loaded {len(docs)} pages from {DOCS_DIR}")
    return docs


def split(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def embed_and_store(chunks):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_postgres import PGVector

    store = PGVector(
        embeddings=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
        connection=get_engine(),
        use_jsonb=True,
        pre_delete_collection=True,  # start clean on every run
    )

    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        for attempt in range(5):
            try:
                store.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < 4:
                    print(f"Rate limited, waiting {PAUSE_SECONDS}s and retrying...")
                    time.sleep(PAUSE_SECONDS)
                else:
                    raise
        done = min(start + BATCH_SIZE, total)
        print(f"Stored {done}/{total} chunks")
        if done < total:
            time.sleep(PAUSE_SECONDS)

    print(f'Finished: {total} chunks in collection "{COLLECTION_NAME}"')


if __name__ == "__main__":
    docs = load_pdfs()
    if not docs:
        sys.exit(f"No PDFs with extractable text found in {DOCS_DIR}")
    embed_and_store(split(docs))
