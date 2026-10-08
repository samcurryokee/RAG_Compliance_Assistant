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
7. Use the terminology of the notes and follow the output format below.
8. If the question is about caring for a patient or choosing a treatment, add one sentence
   saying this is a study aid and to verify with a qualified clinician and current guidelines.

{history}Output format:
{format_instructions}

Context:
{context}

Question: {question}

Answer:"""

# ---- output styles ----------------------------------------------------------------------
STYLES = ("explain", "table", "mcq")
TOP_K_STRUCTURED = int(os.getenv("TOP_K_STRUCTURED", "8"))  # tables and quizzes need more material

FORMAT_INSTRUCTIONS = {
    "explain": "Answer concisely in short paragraphs or bullet points.",
    "table": (
        "Present the answer as ONE Markdown table with a header row and a separator row, "
        "for example:\n"
        "| Feature | Item A | Item B | Source |\n"
        "|---|---|---|---|\n"
        "Choose columns that fit the question, keep cells short and in plain text, and make the "
        "last column 'Source' with the citation for that row. You may add one short sentence "
        "before the table and, if needed, one sentence after it about anything the notes do not "
        "cover. Output nothing else."
    ),
    "mcq": (
        "Write {count} multiple-choice questions based ONLY on the passages, using exactly this "
        "template for every question, with a blank line between questions:\n"
        "Q1. <question text>\n"
        "A) <option>\n"
        "B) <option>\n"
        "C) <option>\n"
        "D) <option>\n"
        "Answer: <one letter>\n"
        "Explanation: <one or two sentences from the notes> [file name, p.N]\n"
        "Rules for the questions: exactly four options and exactly one correct option; the question, "
        "the correct option and the explanation must come from the passages; wrong options may use "
        "plausible terms but must be clearly wrong according to the passages; vary the position of "
        "the correct letter; never use 'all of the above' or 'none of the above'. If the passages "
        "only support fewer questions, write fewer. Optionally start with one short sentence; "
        "output nothing after the last explanation."
    ),
}

# "<file>.pdf, p.12" or "<file>.pdf, p.12-14", inside any bracket style or none.
# Anchored on ".pdf" because file names can contain commas and parentheses.
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


# ---- formatting helpers ------------------------------------------------------------------

TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
FOLLOWUP_RE = re.compile(
    r"\b(that|this|it|its|those|these|them|above|previous|same|more|again|further|also)\b", re.I
)
_MCQ_WORDS = re.compile(r"\b(mcqs?|multiple[- ]choice|quiz|test me|practice questions?)\b", re.I)
_TABLE_WORDS = re.compile(r"\b(table|tabulate|tabulated|compare|comparison|versus|vs\.?)\b", re.I)
_COUNT_RE = re.compile(r"\b(\d{1,2})\s*(?:mcqs?|multiple[- ]choice|questions?|quiz)", re.I)


def detect_style(question):
    """Pick an output style from the wording of the question (used when style is 'auto')."""
    if _MCQ_WORDS.search(question):
        return "mcq"
    if _TABLE_WORDS.search(question):
        return "table"
    return "explain"


def detect_count(question, default=5):
    m = _COUNT_RE.search(question)
    return max(1, min(10, int(m.group(1)))) if m else default


def has_table(text):
    lines = text.splitlines()
    return any(
        TABLE_SEP_RE.match(lines[i]) and "|" in lines[i - 1] for i in range(1, len(lines))
    )


_Q_RE = re.compile(r"^(?:question\s*)?q?\s*(\d{1,2})\s*[.):]\s*(.+)$", re.I)
_OPT_RE = re.compile(r"^\(?([A-Ea-e])\s*[.)]\s+(.+)$")
_ANS_RE = re.compile(
    r"^(?:correct\s+)?answer\s*(?:is\s*)?[:\-]?\s*\(?([A-Ea-e])(?=[\s.):,]|$)", re.I
)
_EXP_RE = re.compile(r"^(?:explanation|rationale|reason|why)\s*[:\-]\s*(.*)$", re.I)


def _strip_md(line):
    line = re.sub(r"^[\s>#*_\-\u2022]+", "", line)
    return line.replace("**", "").replace("__", "").strip()


def parse_quiz(text, count=10):
    """Parse the MCQ template into [{number, question, options:[{letter,text}], answer, explanation}].

    Tolerates markdown bold, '1)' or 'Question 1:' numbering, '(a)' options and a missing
    explanation. Questions that cannot be parsed cleanly are dropped; [] means fall back to text.
    """
    questions, cur, mode = [], None, None

    def finish():
        if not cur:
            return
        letters = [o["letter"] for o in cur["options"]]
        if cur["question"] and len(letters) >= 2 and cur["answer"] in letters:
            questions.append(cur)

    for raw in text.splitlines():
        line = _strip_md(raw)
        if not line:
            continue
        m = _Q_RE.match(line)
        if m and not _OPT_RE.match(line):
            finish()
            cur = {"question": m.group(2).strip(), "options": [], "answer": "", "explanation": ""}
            mode = "question"
            continue
        if cur is None:
            continue
        m = _ANS_RE.match(line)
        if m:
            cur["answer"] = m.group(1).upper()
            mode = None
            continue
        m = _EXP_RE.match(line)
        if m:
            cur["explanation"] = m.group(1).strip()
            mode = "explanation"
            continue
        m = _OPT_RE.match(line)
        if m and mode in ("question", "options"):
            cur["options"].append({"letter": m.group(1).upper(), "text": m.group(2).strip()})
            mode = "options"
            continue
        if mode == "explanation":
            cur["explanation"] = (cur["explanation"] + " " + line).strip()
        elif mode == "question":
            cur["question"] = (cur["question"] + " " + line).strip()
    finish()

    questions = questions[:count]
    for i, q in enumerate(questions, start=1):
        q["number"] = i
    return questions


def history_block(history):
    """Last two Q&A pairs as context for follow-up questions ('' when there are none)."""
    if not history:
        return ""
    lines = ["Conversation so far (context only; answer from the passages below, not from earlier answers):"]
    for h in history[-2:]:
        lines.append(f"User: {h['question'].strip()}")
        lines.append(f"Assistant: {h['answer'].strip()[:600]}")
    return "\n".join(lines) + "\n\n"


def retrieval_query(question, history):
    """Short follow-ups ('make that a table') borrow the previous question for retrieval."""
    if history and len(question.split()) <= 12 and FOLLOWUP_RE.search(question):
        return f"{history[-1]['question']} {question}"
    return question


# ---- main entry points -------------------------------------------------------------------

def ask(question, subject=None, style="auto", count=5, history=None):
    """Answer a question from the notes.

    subject: restrict the search to one subject.  style: auto | explain | table | mcq.
    count: number of MCQs.  history: previous [{'question','answer'}] pairs for follow-ups.
    """
    history = history or []
    if style not in STYLES:
        style = detect_style(question)
        count = detect_count(question, count)
    flt = {"subject": {"$eq": subject}} if subject else None
    k = TOP_K_STRUCTURED if style in ("table", "mcq") else TOP_K
    docs = get_store().similarity_search(retrieval_query(question, history), k=k, filter=flt)

    answer = get_chain().invoke(
        {
            "context": format_docs(docs),
            "question": question,
            "history": history_block(history),
            "format_instructions": FORMAT_INSTRUCTIONS[style].replace("{count}", str(count)),
        }
    ).strip()
    answer = answer.replace("\u3010", "[").replace("\u3011", "]")  # tidy 【 】 into [ ]
    cited = cited_pairs(answer, {source_label(d) for d in docs})

    refused = is_refusal(answer)
    notes, quiz, shown_style = [], [], style
    if refused:
        shown_style = "explain"
    elif style == "mcq":
        quiz = parse_quiz(answer, count)
        if not quiz:
            shown_style = "explain"
            notes.append("Could not format the questions as a quiz, so they are shown as text.")
    elif style == "table" and not has_table(answer):
        shown_style = "explain"
        notes.append("Could not format the answer as a table, so it is shown as text.")

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
        "style": shown_style,
        "quiz": quiz,
        "notes": notes,
        "sources": sources,
        "refused": refused,
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
