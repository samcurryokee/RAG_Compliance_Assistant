"""Retrieval + answer chain for the Kenyan Constitution assistant.

Imported by api.py, ingest.py and the eval scripts. Also runnable as a terminal chat:
    python3 chain.py

Heavy imports (LangChain, SQLAlchemy) happen lazily inside the get_* functions so the
API starts fast and /health responds even if a provider is misconfigured.
"""
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kenyan_constitution")
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL = "models/gemini-embedding-001"  # must match the model used in ingest.py
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
REFUSAL_MARKER = "could not find this in the provided documents"

PROMPT_TEMPLATE = """You are a legal research assistant for the Constitution of Kenya.
Answer the question using ONLY the context below.

Rules:
1. Say "I could not find this in the provided documents." ONLY if none of the passages
   is relevant to the question. If any passage is related, answer from it (see rule 5).
2. Cite every claim as [filename, p.N], copying the label shown above the passage EXACTLY.
   Example: [kenyan_constitution.pdf, p.25]
3. N is always the PAGE number from the label. Never use an Article number as a page number.
4. Only cite pages whose labels appear in the context below.
5. If the passages address the topic only indirectly, do NOT refuse. Say what the text
   does not state, then explain what the relevant passage does say and cite it.
   Example pattern: "The provided text does not mention X by name. However, [provision]
   says ... [citation]."
6. For yes/no questions, answer "yes" or "no" only if the passages directly answer it.
   Otherwise use the pattern in rule 5.
7. Do not conclude beyond what the passages state. If related provisions such as
   exceptions or emergency rules might exist but were not provided, say so.
8. Do not add qualifiers such as "expressly" or "explicitly" unless those words
   appear in the passage.
9. Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""


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
        database_url(),
        pool_size=2,
        max_overflow=2,
        pool_pre_ping=True,
        pool_recycle=300,
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


def page_label(doc):
    # PDF pages are stored 0-indexed; +1 so it matches the page in a PDF viewer
    return int(doc.metadata.get("page", 0)) + 1


def source_label(doc):
    return os.path.basename(doc.metadata.get("source", "unknown"))


def format_docs(docs):
    return "\n\n".join(
        f"[{source_label(d)}, p.{page_label(d)}]\n{d.page_content}" for d in docs
    )


def cited_pages(answer):
    """Set of page numbers the answer cites, e.g. {25, 26}."""
    return {int(n) for n in re.findall(r"p\.\s*(\d+)", answer)}


def check_citations(answer, docs):
    """Cited pages that were NOT among the retrieved pages (likely hallucinated)."""
    retrieved = {page_label(d) for d in docs}
    return sorted(cited_pages(answer) - retrieved)


def is_refusal(answer):
    return REFUSAL_MARKER in answer.lower()


def make_snippet(text, limit=350):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def ask(question):
    docs = get_store().similarity_search(question, k=TOP_K)
    answer = get_chain().invoke(
        {"context": format_docs(docs), "question": question}
    ).strip()
    cited = cited_pages(answer)

    sources, seen = [], set()
    for d in docs:  # retrieval order, one entry per page
        key = (source_label(d), page_label(d))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "file": key[0],
                "page": key[1],
                "snippet": make_snippet(d.page_content),
                "cited": key[1] in cited,
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "refused": is_refusal(answer),
        "bad_citations": check_citations(answer, docs),
        "retrieved_pages": [page_label(d) for d in docs],
    }


if __name__ == "__main__":
    print("Kenya Constitution assistant. Type 'quit' to exit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        result = ask(q)
        print(f"\n{result['answer']}\n")
        print("Sources retrieved:")
        for s in result["sources"]:
            mark = " (cited)" if s["cited"] else ""
            print(f"  - {s['file']}, p.{s['page']}{mark}")
        if result["bad_citations"]:
            pages = ", ".join(f"p.{p}" for p in result["bad_citations"])
            print(f"\nWARNING: answer cites pages that were not retrieved: {pages}")
        print()
