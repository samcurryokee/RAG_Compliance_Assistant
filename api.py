"""FastAPI service for the medical notes assistant.

Run locally:   uvicorn api:app --reload
Endpoints:     GET /health[?db=1]   GET /documents   POST /ask
"""
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

import chain  # noqa: E402  (after load_dotenv so env vars are set)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",")
    if o.strip()
]
ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALLOWED_ORIGIN_REGEX", r"https://.*\.(lovable\.app|lovableproject\.com)"
)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "6"))
DAILY_CAP = int(os.getenv("DAILY_CAP", "300"))
DOCS_CACHE_SECONDS = 300


class RateLimiter:
    """In-memory limiter: per-IP per-minute plus a global daily cap.

    Resets when the free Render instance restarts, which is acceptable for a demo.
    The global daily cap is the real protection for the free Gemini/Groq quotas.
    """

    def __init__(self, per_minute, daily_cap):
        self.per_minute = per_minute
        self.daily_cap = daily_cap
        self.hits = defaultdict(deque)
        self.day = date.today()
        self.count = 0
        self.lock = threading.Lock()

    def check(self, ip):
        now = time.time()
        with self.lock:
            today = date.today()
            if today != self.day:
                self.day, self.count = today, 0
            if self.count >= self.daily_cap:
                return "daily"
            if len(self.hits) > 5000:  # keep memory bounded
                self.hits = defaultdict(
                    deque, {k: v for k, v in self.hits.items() if v and now - v[-1] < 60}
                )
            q = self.hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return "minute"
            q.append(now)
            self.count += 1
            return None


limiter = RateLimiter(RATE_LIMIT_PER_MINUTE, DAILY_CAP)
_docs_cache = {"time": 0.0, "data": None}
_docs_lock = threading.Lock()

app = FastAPI(title="Medical Notes Assistant", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    subject: Optional[str] = Field(None, max_length=60)


class Source(BaseModel):
    file: str
    title: str
    subject: str
    page: int
    page_end: int
    snippet: str
    cited: bool


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    warnings: list[str] = []


class DocumentInfo(BaseModel):
    file: str
    title: str
    subject: str
    pages: int
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    subjects: list[str]
    total_pages: int
    total_chunks: int


def client_ip(request: Request):
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_documents():
    """The embedded documents, cached for a few minutes to spare the database."""
    with _docs_lock:
        if _docs_cache["data"] is None or time.time() - _docs_cache["time"] > DOCS_CACHE_SECONDS:
            _docs_cache["data"] = chain.list_documents()
            _docs_cache["time"] = time.time()
        return _docs_cache["data"]


@app.get("/")
def root():
    return {"service": "Medical Notes Assistant", "endpoints": ["/health", "/documents", "/ask"]}


@app.get("/health")
def health(db: int = 0):
    """Cheap liveness check. /health?db=1 also runs SELECT 1 (used by the keep-alive ping)."""
    if db:
        try:
            from sqlalchemy import text

            with chain.get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            log.exception("database health check failed")
            raise HTTPException(status_code=503, detail="Database unreachable")
    return {"status": "ok", "db_checked": bool(db)}


@app.get("/documents", response_model=DocumentsResponse)
def documents():
    """The notes this assistant can search, so users know exactly what its answers are based on."""
    try:
        docs = get_documents()
    except Exception:
        log.exception("listing documents failed")
        raise HTTPException(503, "Could not load the document list right now.")
    return {
        "documents": docs,
        "subjects": sorted({d["subject"] for d in docs}),
        "total_pages": sum(d["pages"] for d in docs),
        "total_chunks": sum(d["chunks"] for d in docs),
    }


# Sync def on purpose: FastAPI runs it in a worker thread, so slow LLM calls
# do not block the event loop.
@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request):
    subject = (body.subject or "").strip() or None
    if subject:
        known = {d["subject"] for d in get_documents()}
        if known and subject not in known:
            raise HTTPException(400, f"Unknown subject. Choose one of: {', '.join(sorted(known))}")

    blocked = limiter.check(client_ip(request))
    if blocked == "minute":
        raise HTTPException(429, "Too many questions. Please wait a minute and try again.")
    if blocked == "daily":
        raise HTTPException(
            429, "The daily question limit for this free demo has been reached. Try again tomorrow."
        )

    try:
        result = chain.ask(body.question.strip(), subject)
    except Exception as e:
        log.exception("ask failed")
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg or "rate limit" in msg.lower():
            raise HTTPException(503, "The AI providers are rate-limited right now. Try again in a minute.")
        raise HTTPException(500, "Something went wrong answering that question.")

    warnings = []
    if result["bad_citations"]:
        warnings.append(
            "The answer cites pages that were not among the retrieved passages "
            f"({', '.join(result['bad_citations'])}). Please verify those citations in the notes."
        )
    return {"answer": result["answer"], "sources": result["sources"], "warnings": warnings}
