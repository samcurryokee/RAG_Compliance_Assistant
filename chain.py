"""Retrieval + answer chain for the medical notes assistant.

Imported by api.py, ingest.py and the eval scripts. Also runnable as a terminal chat:
    python3 chain.py [subject]        e.g. python3 chain.py Neuroanatomy

Heavy imports (LangChain, SQLAlchemy) happen lazily inside the get_* functions so the API
starts fast and /health responds even if a provider is misconfigured.
"""
import os
import re
import sys
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_notes")
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL = "models/gemini-embedding-001"  # must match the model used in ingest.py
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
REFUSAL_MARKER = "could not find this in the provided documents"

PROMPT_TEMPLATE = """You are a study assistant for medical students. Answer the question using ONLY the lecture notes in the context below.

Rules:
1. Say "I could not find this in the provided documents." ONLY if none of the passages
   is relevant to the question. If any passage is related, answer from it (see rule 5).
2. Cite every claim as [filename, p.N], copying the label shown above the passage EXACTLY.
    If the label shows a page range, cite it as shown. Always use plain square brackets [ ],
   never any other bracket style, and one citation per pair of brackets.
   Example: [Basic Embryology, Pulei.pdf, p.40-42]
3. Only cite labels that appear in the context below. Never invent a file or page.
4. The notes come from slides and OCR text and may contain typos or garbled words.
   If a passage looks garbled or incomplete, say so instead of guessing.
5. If the passages address the topic only partly, do NOT refuse. Say what the notes do
   not state, then explain what the relevant passages do say and cite them.
6. Do not add facts, numbers or clinical advice that are not in the passages.
   Do not use outside knowledge.
7. Keep answers concise and structured (short paragraphs or bullet points) and use the
   terminology of the notes.
8. If the question is about caring for a patient or choosing a treatment, add one sentence
   saying this is a study aid and to verify with a qualified clinician and current guidelines.

Context:
{context}

Question: {question}

Answer:"""

# [file name, p.12] or [file name, p.12-14]; file names may contain commas
CITATION_RE = re.compile(
    r"([^\[\]\u3010\u3011\n]*?\.pdf)\s*,\s*pp?\.\s*(\d+)(?:\s*[-\u2013\u2014]\s*(\d+))?",
    re.IGNORECASE,
)


def database_url():
    """DATABASE_URL with the psycopg3 driver prefix SQLAlchemy needs."""
    url = os.environ["DATABASE_URL"].strip()
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


@lru_cache(maxsize=1)
def get_engine():
    from sqlalchemy import create_engine

    return create_engine(
        database_url(), pool_size=2, max_overflow=2, pool_pre_ping=True, pool_recycle=300
    )


@lru_cache(maxsize=1)
def get_store():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_postgres import PGVector

    # No pre_delete_collection here: that would wipe the ingested data.
    return PGVector(
        embeddings=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
        connection=get_engine(),
        use_jsonb=True,
    )


@lru_cache(maxsize=1)
def get_chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != GROQ_MODEL:
        llm = llm.with_fallbacks([ChatGroq(model=GROQ_FALLBACK_MODEL, temperature=0)])
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | llm | StrOutputParser()


# ---- page labels -------------------------------------------------------------------------
# Chunks store page (first page, 0-indexed) and page_end (last page, 0-indexed) because
# several short slides are merged into one chunk. Labels are 1-based, as in a PDF viewer.

def page_span(doc):
    start = int(doc.metadata.get("page", 0)) + 1
    end = int(doc.metadata.get("page_end", doc.metadata.get("page", 0))) + 1
    return start, max(start, end)


def page_text(start, end):
    return f"{start}" if start == end else f"{start}-{end}"


def source_label(doc):
    return os.path.basename(doc.metadata.get("source", "unknown"))


def doc_label(doc):
    return f"[{source_label(doc)}, p.{page_text(*page_span(doc))}]"


def format_docs(docs):
    return "\n\n".join(f"{doc_label(d)}\n{d.page_content}" for d in docs)


# ---- citation checks ---------------------------------------------------------------------

def _clean_name(raw, known):
    """Strip bracket/punctuation junk, and any words before a known file name."""
    raw = raw.strip(" \t([\u3010;,:")
    for k in sorted(known, key=len, reverse=True):
        if raw.endswith(k):
            return k
    return raw


def cited_pairs(answer, known_files=()):
    """Set of (file, page) pairs the answer cites; page ranges are expanded.

    known_files: file names to match against, so "see X.pdf, p.3" (no brackets) still resolves to X.pdf.
    """
    pairs = set()
    for name, a, b in CITATION_RE.findall(answer):
        start = int(a)
        end = int(b) if b else start
        if end < start or end - start > 60:
            end = start
        for p in range(start, end + 1):
            pairs.add((_clean_name(name, known_files), p))
    return pairs


def retrieved_pairs(docs):
    pairs = set()
    for d in docs:
        start, end = page_span(d)
        for p in range(start, end + 1):
            pairs.add((source_label(d), p))
    return pairs


def check_citations(answer, docs):
    """Citations (as 'file, p.N') that do NOT match any retrieved (file, page): likely invented."""
    known = {source_label(d) for d in docs}
    bad = sorted(cited_pairs(answer, known) - retrieved_pairs(docs))
    return [f"{f}, p.{p}" for f, p in bad]


def is_refusal(answer):
    return REFUSAL_MARKER in answer.lower()


def make_snippet(text, limit=350):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


# ---- main entry points -------------------------------------------------------------------

def ask(question, subject=None):
    """Answer a question from the notes. subject optionally restricts the search."""
    flt = {"subject": {"$eq": subject}} if subject else None
    docs = get_store().similarity_search(question, k=TOP_K, filter=flt)
    answer = get_chain().invoke({"context": format_docs(docs), "question": question}).strip()
    answer = answer.replace("\u3010", "[").replace("\u3011", "]")  # tidy 【 】 into [ ]
    cited = cited_pairs(answer, {source_label(d) for d in docs})

    sources, seen = [], set()
    for d in docs:  # retrieval order, one entry per page span
        start, end = page_span(d)
        key = (source_label(d), start, end)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "file": key[0],
                "title": d.metadata.get("title", key[0]),
                "subject": d.metadata.get("subject", ""),
                "page": start,
                "page_end": end,
                "snippet": make_snippet(d.page_content),
                "cited": any((key[0], p) in cited for p in range(start, end + 1)),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "refused": is_refusal(answer),
        "cited": cited,
        "bad_citations": check_citations(answer, docs),
        "retrieved": [(s["file"], s["page"], s["page_end"]) for s in sources],
    }


def list_documents():
    """Documents that are actually embedded, from the manifest table written by ingest.py."""
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError

    sql = text(
        "SELECT file, title, subject, pages, chunks FROM rag_documents "
        "WHERE collection = :c ORDER BY subject, title"
    )
    try:
        with get_engine().connect() as conn:
            return [dict(r) for r in conn.execute(sql, {"c": COLLECTION_NAME}).mappings().all()]
    except ProgrammingError:  # table does not exist yet (nothing ingested)
        return []


if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Medical notes assistant{' (' + subject + ')' if subject else ''}. Type 'quit' to exit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        result = ask(q, subject)
        print(f"\n{result['answer']}\n")
        print("Sources retrieved:")
        for s in result["sources"]:
            mark = " (cited)" if s["cited"] else ""
            print(f"  - {s['title']} [{s['file']}], p.{page_text(s['page'], s['page_end'])}{mark}")
        if result["bad_citations"]:
            print(f"\nWARNING: citations not among the retrieved passages: {', '.join(result['bad_citations'])}")
        print()
