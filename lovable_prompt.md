# Prompt to paste into Lovable

Replace `YOUR-SERVICE` in the API URL with your real Render service name first.

---

Build a single-page web app called "Kenya Constitution Assistant": a chat-style interface
for asking questions about the Constitution of Kenya, answered with page citations.

HARD CONSTRAINTS
- Front end only. Do NOT enable Lovable Cloud, Supabase, any database, authentication,
  edge functions or any other backend. Store nothing. No login, no analytics.
- All data comes from one external REST API, called with fetch from the browser.
- Put the API base URL in one constant in `src/lib/config.ts`:
  `export const API_BASE_URL = "https://YOUR-SERVICE.onrender.com";`

API CONTRACT
POST `${API_BASE_URL}/ask` with JSON `{ "question": string }` (3 to 500 characters).
Success (200):
```json
{
  "answer": "string with [kenyan_constitution.pdf, p.25] style citations",
  "sources": [
    { "file": "kenyan_constitution.pdf", "page": 25, "snippet": "text...", "cited": true }
  ],
  "warnings": ["string"]
}
```
Errors return `{ "detail": "message" }` with status 429 (rate limited), 503 (providers busy)
or 500. Show the `detail` text to the user in a friendly error box.

BEHAVIOUR
- One question box with a Send button. Disable it while a request is running.
- Show 4 clickable example questions that fill the box and submit:
  "What does the constitution say about freedom of expression?",
  "What rights does an arrested person have?",
  "How can a President be removed from office?",
  "What is the right to privacy?"
- The server is on a free plan and sleeps when idle. If a request takes longer than 5
  seconds, show "Waking up the free server, the first question can take up to a minute."
  Use a 90 second timeout, then show a retry message.
- Render the answer as plain text that preserves line breaks and **bold**.
- Below the answer show a "Sources" section with one card per source: file name, page
  number, and the snippet. Sources with `cited: true` get a "Cited in answer" badge and
  are listed first; the rest go under "Also retrieved".
- If `warnings` is not empty, show it in a small yellow notice above the sources.
- Keep the conversation history on screen (question and answer pairs) during the session.
- Footer: "Answers are generated from the text of the Constitution and may contain errors.
  Verify against the source. Not legal advice."

DESIGN
Clean, minimal, mobile-friendly, with light and dark mode. Calm colours, readable type,
generous spacing.
