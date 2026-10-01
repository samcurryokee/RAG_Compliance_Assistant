import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
import time

BATCH_SIZE = 80      # stay under the 100 requests/minute free-tier limit
PAUSE_SECONDS = 65   # let the per-minute quota reset between batches
load_dotenv()

DOCS_DIR = "data/documents"
COLLECTION_NAME = "kenyan_constitution"




def load_and_split():
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    raw_docs = loader.load()  # one Document per PDF page, with .metadata['source'] and ['page']
    print(f"Loaded {len(raw_docs)} pages from {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks





def embed_and_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=os.environ["DATABASE_URL"],
        use_jsonb=True,
        pre_delete_collection=True,  # start clean on every run
    )

    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        for attempt in range(5):
            try:
                vectorstore.add_documents(batch)
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
    return vectorstore


if __name__ == "__main__":
    chunks = load_and_split()
    embed_and_store(chunks)