# Medical Notes Assistant (RAG)

Ask questions about a set of anatomy, embryology and neuroscience lecture notes. The assistant
retrieves the relevant passages, answers **only** from them, and cites the file and page for
every claim. The app also shows which documents it is built on (`GET /documents`).

**Stack:** LangChain, pgvector (Postgres), Gemini embeddings, Groq LLM, FastAPI, Tesseract OCR,
Lovable front end. Every service runs on a free tier.

```
Lovable (React chat UI)
      |  POST /ask    GET /documents
      v
FastAPI on Render (free web service)
      |                    |                    |
      v                    v                    v
Supabase Postgres     Gemini API           Groq API
(pgvector)            (query embeddings)   (answers)
```

## Documents used

| File | Subject | Pages |
|---|---|---|
| Basic Embryology, Pulei.pdf | Embryology | 373 |
| ALL CONGENITAL ANOMALIES AND THEIR BASIS (2).pdf | Embryology | 50 |
| Neuroanat Review Session Notes.pdf | Neuroanatomy | 327 |
| Neuroanatomy Lecture Notes Final.pdf | Neuroanatomy | 1033 |
| Neurohistology Lecture Notes Final (1).pdf | Neurohistology | 270 |

**Not included:** Embryo Dan Series (157 pages) and the five "Objectives Answered" files
(Head and Neck, Lower Limb, Neuroanatomy, Thorax and Abdomen, Upper Limb; 392 pages). They are
photos of handwriting, which Tesseract cannot read reliably. As a result, gross anatomy of the
head, neck, limbs, thorax and abdomen is not covered, and questions about it will be refused.

Titles and subjects come from `documents.json`. Add an entry there when you add a PDF.
The live list is served by `GET /documents`, read from the `rag_documents` table that
`ingest.py` fills, so it always matches what is actually searchable.

## Repo layout

| File | Purpose |
|---|---|
| `Dockerfile.extract` | Optional: runs `extract.py` in Docker, no Tesseract install needed |
| `extract.py` | Laptop only. PDF pages to text (text layer, else Tesseract OCR); resumable; flags handwriting |
| `ingest.py` | Laptop only. Chunks the extracted text, embeds with Gemini, stores in pgvector; incremental |
| `chain.py` | Retrieval, prompt, Groq answer, (file, page) citation check; also a terminal chat |
| `api.py` | FastAPI: `POST /ask`, `GET /documents`, `GET /health`, CORS, rate limiting |
| `eval_retrieval.py` | Hit rate @3/@5 and MRR against `golden_set.json` |
| `eval_answers.py` | Full-chain checks (refusals, citations, expected pages); logs to `eval_history.jsonl` |
| `find_pages.py` | Search the extracted text, to help write golden questions |
| `documents.json` | Title and subject for each PDF |
| `docker-compose.yml` | Local Postgres with pgvector |
| `render.yaml` | Render deployment blueprint |
| `.github/workflows/keepalive.yml` | Daily ping so free Supabase does not pause |
| `lovable_prompt.md` | Prompt that generates the front end in Lovable |

## Run locally

```bash
brew install tesseract                      # macOS; needed for OCR
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-ingest.txt
cp .env.example .env                        # fill in GOOGLE_API_KEY, GROQ_API_KEY, DATABASE_URL
docker compose up -d                        # local Postgres (or point DATABASE_URL at Supabase)
# put the five PDFs in data/documents/
python3 extract.py --workers 4              # OCR; slowest step, resumable, safe to stop and restart
python3 ingest.py                           # embeds only new or changed files
python3 chain.py                            # chat in the terminal (python3 chain.py Neuroanatomy to filter)
python3 -m uvicorn api:app --reload         # API at http://localhost:8000/docs
```

Expect `extract.py` to take a while (slides are OCR'd at roughly 1 to 2 seconds a page per
worker). Embedding is paced to the Gemini free-tier limit of 100 requests a minute.
`extract.py` prints a warning for any file whose OCR confidence is low, and `ingest.py` skips it.

### No Tesseract on your Mac? Extract with Docker instead

On an Intel Mac, `brew install tesseract` can compile for a long time. Docker avoids that:

```bash
docker build -f Dockerfile.extract -t notes-extract .
docker run --rm -v "$PWD/data:/app/data" notes-extract --workers 4
python3 ingest.py          # on the Mac as usual; it does not need Tesseract
```

The container reads `data/documents/` and writes `data/extracted/` through the mounted folder, and
it is resumable like the normal run. Set `--workers` to about your CPU cores minus one, and in
Docker Desktop (Settings > Resources) give Docker at least that many CPUs, since it defaults to
fewer than the Mac has. On the Mac itself you then only need `pip install -r requirements.txt`
plus `langchain-text-splitters`.

## Evaluate

1. `python3 eval_retrieval.py` for retrieval quality.
2. `python3 eval_answers.py baseline` for the full chain. After any change (prompt, chunking, k,
   model) run it again with a new label and compare the lines in `eval_history.jsonl`.

The questions in `golden_set.json` were written from pages of these notes. Add your own, with
the file and pages where the answer lives (`python3 find_pages.py "phrase"` finds them).

## Deploy for free

1. **Supabase:** create a free project, copy the **Session pooler** connection string from the
   Connect panel into `.env` as `DATABASE_URL` (URL-encode special characters in the password),
   then run `python3 ingest.py` from your laptop.
2. **GitHub:** push this repo. `.gitignore` keeps `.env`, the PDFs and the extracted text out of it.
3. **Render:** New > Blueprint, pick the repo, enter `GOOGLE_API_KEY`, `GROQ_API_KEY` and
   `DATABASE_URL` when prompted. Check `https://YOUR-SERVICE.onrender.com/health?db=1` and `/documents`.
4. **Lovable:** paste `lovable_prompt.md` with your Render URL filled in. Then set
   `ALLOWED_ORIGIN_REGEX` on Render to your exact Lovable domain.
5. **Keep-alive:** GitHub repo > Settings > Secrets and variables > Actions > Variables:
   add `API_URL` = your Render URL.

## Free-tier caveats

- Render free services sleep after about 15 minutes idle; the first request then takes around a minute.
- Free Supabase projects pause after a week of inactivity (the keep-alive workflow helps).
- Gemini and Groq free tiers have rate limits that change. `api.py` adds a per-IP and a daily cap
  (`RATE_LIMIT_PER_MINUTE`, `DAILY_CAP`) so strangers cannot use up your quota.
- API keys live only in server environment variables, never in the front end.

## Known limitations

- Slides are OCR'd, so text can contain errors, and **diagrams are not searchable**: only their
  labels are captured. Questions about a figure will answer poorly.
- Answers can still be wrong. The citation check only verifies that cited (file, page) pairs were
  retrieved, not that the cited text supports the claim.
- This is a study aid, not clinical advice. Verify against the source notes and current guidelines.
