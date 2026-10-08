# Prompt to paste into Lovable

Replace `MEDICAL NOTES ASSISTANT` in the API URL with your real Render service name first.

---

Build a single-page web app called "Medical Notes Assistant": a chat-style study tool for asking
questions about a fixed set of lecture notes (embryology, neuroanatomy, neurohistology), answered
with file and page citations. It must also show which documents the answers are based on.

HARD CONSTRAINTS
- Front end only. Do NOT enable Lovable Cloud, Supabase, any database, authentication, edge
  functions or any other backend. Store nothing. No login, no analytics.
- All data comes from one external REST API, called with fetch from the browser.
- Put the API base URL in one constant in `src/lib/config.ts`:
  `export const API_BASE_URL = "https://medical-notes-api.onrender.com";`

API CONTRACT
1) GET `${API_BASE_URL}/documents` returns:
```json
{
  "documents": [
    { "file": "Basic Embryology, Pulei.pdf", "title": "Basic Embryology (Dr. Pulei)",
      "subject": "Embryology", "pages": 373, "chunks": 210 }
  ],
  "subjects": ["Embryology", "Neuroanatomy", "Neurohistology"],
  "total_pages": 2053,
  "total_chunks": 900
}
```
2) POST `${API_BASE_URL}/ask` with JSON `{ "question": string, "subject": string | null }`
(question 3 to 500 characters; send `subject` only when the user picked one). Returns:
```json
{
  "answer": "text with [Basic Embryology, Pulei.pdf, p.130] style citations",
  "sources": [
    { "file": "Basic Embryology, Pulei.pdf", "title": "Basic Embryology (Dr. Pulei)",
      "subject": "Embryology", "page": 130, "page_end": 131,
      "snippet": "text...", "cited": true }
  ],
  "warnings": ["string"]
}
```
Errors return `{ "detail": "message" }` with status 400, 429 (rate limited), 503 (providers busy)
or 500. Show the `detail` text to the user in a friendly error box.

LAYOUT AND BEHAVIOUR
- "Notes library" panel (a sidebar on desktop, a collapsible section on mobile). On load, call
  GET /documents and list the documents grouped by subject, showing each title, the file name in
  small grey text, and its page count. Above the list show the totals, e.g. "5 documents, 2,053
  pages". If the call fails, show "Could not load the document list" and keep the chat usable.
- Subject filter: chips for "All subjects" plus one per subject from the API. The chosen subject is
  sent with each question. Default is all subjects.
- One question box with a Send button, disabled while a request runs.
- Show 4 clickable example questions that fill the box and submit:
  "What is a morula and when does it enter the uterus?",
  "What is the difference between meningocele and myelomeningocele?",
  "What are the types of white matter fibres?",
  "What are the six layers of the cerebral cortex?"
- The server is on a free plan and sleeps when idle. If a request takes longer than 5 seconds, show
  "Waking up the free server, the first question can take up to a minute." Use a 90 second timeout,
  then show a retry message.
- Render the answer as plain text that preserves line breaks, **bold** and bullet lists.
- Below each answer show "Sources", one card per source: the document title, the file name, the page
  (show "p.130" when page equals page_end, otherwise "pp.130-131"), and the snippet. Sources with
  `cited: true` get a "Cited in answer" badge and come first; the rest go under "Also retrieved".
- If `warnings` is not empty, show it in a small yellow notice above the sources.
- Keep the conversation (question and answer pairs) on screen during the session.
- Footer: "Study aid built from the notes listed in the library. Answers may contain errors, and
  diagrams are not searchable. Verify against the source notes. Not clinical advice."

DESIGN
Clean, minimal, mobile-friendly, with light and dark mode. Calm colours, readable type, generous spacing.
