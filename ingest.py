"""Embed the extracted notes into pgvector (incremental).

Usage:
  python3 extract.py                    # once: read/OCR the PDFs into data/extracted/
  python3 ingest.py                     # embed new or changed files, skip the rest
  python3 ingest.py --rebuild           # wipe the collection and embed everything again
  python3 ingest.py --prune             # also remove files that are no longer in data/documents/
  python3 ingest.py --only Embryo       # only files whose name contains "Embryo"

Also records every embedded file in the rag_documents table. The API's GET /documents
reads that table, so the document list users see is exactly what is searchable.
Batched and paced to stay under the Gemini free-tier limit (100 requests/minute).
"""
import argparse
import hashlib
import json
import os
import sys
import time

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import text

load_dotenv()

from chain import COLLECTION_NAME, EMBEDDING_MODEL, get_engine  # noqa: E402
from extract import (  # noqa: E402
    DOCS_DIR, cache_path, clean_pages, file_quality, is_complete, is_low_quality, list_pdfs,
    load_extracted, sha256_file,
)

BATCH_SIZE = 80
PAUSE_SECONDS = 65
MIN_BLOCK_CHARS = 500   # merge consecutive short slides until a block reaches this size
MAX_BLOCK_PAGES = 4

MANIFEST_DDL = text("""
CREATE TABLE IF NOT EXISTS rag_documents (
    collection  text        NOT NULL,
    file        text        NOT NULL,
    title       text        NOT NULL,
    subject     text        NOT NULL,
    fingerprint text        NOT NULL,
    pages       integer     NOT NULL,
    chunks      integer     NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, file)
)
""")
UPSERT_SQL = text("""
INSERT INTO rag_documents (collection, file, title, subject, fingerprint, pages, chunks)
VALUES (:c, :f, :t, :s, :fp, :p, :n)
ON CONFLICT (collection, file) DO UPDATE SET
    title = EXCLUDED.title, subject = EXCLUDED.subject, fingerprint = EXCLUDED.fingerprint,
    pages = EXCLUDED.pages, chunks = EXCLUDED.chunks, ingested_at = now()
""")
DELETE_CHUNKS_SQL = text("""
DELETE FROM langchain_pg_embedding e
USING langchain_pg_collection c
WHERE e.collection_id = c.uuid AND c.name = :c AND e.cmetadata->>'source' = :f
""")


def load_catalog():
    """documents.json -> {file name: (title, subject)}"""
    path = "documents.json"
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {d["file"]: (d["title"], d["subject"]) for d in json.load(f)}


def build_blocks(pages):
    """Merge runs of short consecutive slides into blocks. pages: [{'page','text'}] sorted."""
    blocks, cur = [], None
    for p in pages:
        if (
            cur is not None
            and len(cur["text"]) < MIN_BLOCK_CHARS
            and cur["n"] < MAX_BLOCK_PAGES
            and p["page"] - cur["end"] <= 2
        ):
            cur["text"] += "\n\n" + p["text"]
            cur["end"], cur["n"] = p["page"], cur["n"] + 1
        else:
            if cur:
                blocks.append(cur)
            cur = {"start": p["page"], "end": p["page"], "text": p["text"], "n": 1}
    if cur:
        blocks.append(cur)
    return blocks


def build_chunks(file, title, subject, pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for b in build_blocks(pages):
        for piece in splitter.split_text(b["text"]):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": file,
                        "title": title,
                        "subject": subject,
                        "page": b["start"] - 1,      # 0-indexed, like the old loader
                        "page_end": b["end"] - 1,
                    },
                )
            )
    return chunks


def embed_batches(store, chunks):
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        for attempt in range(5):
            try:
                store.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < 4:
                    print(f"    rate limited, waiting {PAUSE_SECONDS}s and retrying...")
                    time.sleep(PAUSE_SECONDS)
                else:
                    raise
        done = min(start + BATCH_SIZE, total)
        print(f"    embedded {done}/{total} chunks")
        if done < total:
            time.sleep(PAUSE_SECONDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="wipe the collection first")
    ap.add_argument("--prune", action="store_true", help="remove files no longer in the documents folder")
    ap.add_argument("--only", help="only files whose name contains this text")
    ap.add_argument("--allow-low-quality", action="store_true",
                    help="also ingest files flagged as handwritten / low OCR confidence")
    args = ap.parse_args()

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_postgres import PGVector

    engine = get_engine()
    catalog = load_catalog()
    pdfs = list_pdfs()
    if not pdfs:
        sys.exit(f"No PDFs found in {DOCS_DIR}")

    # check every extraction cache before spending any embedding quota
    jobs = []
    for pdf in pdfs:
        name = os.path.basename(pdf)
        if args.only and args.only.lower() not in name.lower():
            continue
        sha = sha256_file(pdf)
        meta, pages = load_extracted(cache_path(pdf))
        if not is_complete(meta, pages, sha):
            sys.exit(f'"{name}" has not been fully extracted. Run: python3 extract.py')
        if is_low_quality(pages) and not args.allow_low_quality:
            conf, _ = file_quality(pages)
            print(f'[{name}] SKIPPED: OCR confidence {conf:.0f}% suggests handwriting. '
                  "Tesseract cannot read it reliably (use --allow-low-quality to force).")
            continue
        title, subject = catalog.get(name, (os.path.splitext(name)[0], "General"))
        fingerprint = hashlib.sha256(f"{sha}|{title}|{subject}".encode()).hexdigest()
        jobs.append((name, title, subject, fingerprint, meta["total_pages"], pages))

    with engine.begin() as conn:
        conn.execute(MANIFEST_DDL)

    store = PGVector(
        embeddings=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
        pre_delete_collection=args.rebuild,
    )
    if args.rebuild:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM rag_documents WHERE collection = :c"), {"c": COLLECTION_NAME})

    with engine.connect() as conn:
        existing = dict(conn.execute(
            text("SELECT file, fingerprint FROM rag_documents WHERE collection = :c"),
            {"c": COLLECTION_NAME},
        ).all())

    for name, title, subject, fingerprint, total_pages, pages in jobs:
        if existing.get(name) == fingerprint:
            print(f"[{name}] up to date, skipping")
            continue
        page_list = clean_pages([pages[k] for k in sorted(pages)])
        chunks = build_chunks(name, title, subject, page_list)
        print(f"[{name}] {len(page_list)} usable pages -> {len(chunks)} chunks ({subject})")
        with engine.begin() as conn:  # replace any older version of this file
            conn.execute(DELETE_CHUNKS_SQL, {"c": COLLECTION_NAME, "f": name})
        embed_batches(store, chunks)
        with engine.begin() as conn:
            conn.execute(UPSERT_SQL, {"c": COLLECTION_NAME, "f": name, "t": title, "s": subject,
                                      "fp": fingerprint, "p": total_pages, "n": len(chunks)})

    if args.prune:
        on_disk = {os.path.basename(p) for p in pdfs}
        for gone in set(existing) - on_disk:
            print(f"[{gone}] no longer on disk, removing")
            with engine.begin() as conn:
                conn.execute(DELETE_CHUNKS_SQL, {"c": COLLECTION_NAME, "f": gone})
                conn.execute(text("DELETE FROM rag_documents WHERE collection = :c AND file = :f"),
                             {"c": COLLECTION_NAME, "f": gone})

    with engine.connect() as conn:
        n_docs, n_chunks = conn.execute(
            text("SELECT count(*), coalesce(sum(chunks), 0) FROM rag_documents WHERE collection = :c"),
            {"c": COLLECTION_NAME},
        ).one()
    print(f'Done. Collection "{COLLECTION_NAME}": {n_docs} documents, {n_chunks} chunks.')


if __name__ == "__main__":
    main()
