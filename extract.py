"""Extract text from every PDF in data/documents/ into data/extracted/<file>.jsonl.

Many lecture PDFs are slides saved as images (no text layer), so each page is read from
its text layer when it has one, and OCR'd with Tesseract otherwise.

Usage:
  python3 extract.py                 # all PDFs, resumable
  python3 extract.py --workers 4     # parallel OCR processes
  python3 extract.py --only Embryo   # only files whose name contains "Embryo"

Safe to stop and re-run: finished pages are kept, and a changed PDF is re-extracted.
Needs Tesseract installed (macOS: brew install tesseract).

Handwriting cannot be read by Tesseract. Each OCR'd page records Tesseract's word confidence;
a file whose average is below LOW_CONFIDENCE is flagged here and skipped by ingest.py.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

DOCS_DIR = os.getenv("DOCS_DIR", "data/documents")
EXTRACT_DIR = os.getenv("EXTRACT_DIR", "data/extracted")
OCR_DPI = 200
MIN_TEXT_LAYER_CHARS = 100   # fewer characters than this on a page -> treat as scanned and OCR
MIN_PAGE_CHARS = 25          # after cleaning, shorter pages are dropped (title/diagram-only)
TASK_PAGES = 10              # pages per worker task (also the resume granularity)
LOW_CONFIDENCE = 60          # mean OCR word confidence below this = probably handwriting / bad scan


def list_pdfs():
    return sorted(glob.glob(os.path.join(DOCS_DIR, "**", "*.pdf"), recursive=True))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cache_path(pdf_path):
    return os.path.join(EXTRACT_DIR, os.path.basename(pdf_path) + ".jsonl")


def load_extracted(path):
    """Return (meta, {page: record}) from a cache file, or (None, {}) if missing/unreadable."""
    if not os.path.exists(path):
        return None, {}
    meta, pages = None, {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line from an interrupted run
            if i == 0:
                meta = rec
            elif "page" in rec:
                pages[rec["page"]] = rec
    return meta, pages


def is_complete(meta, pages, sha):
    return bool(meta) and meta.get("sha256") == sha and len(pages) >= meta.get("total_pages", -1)


def clean_pages(pages):
    """Drop boilerplate (lines repeated on many pages, bare slide numbers) and near-empty pages.

    pages: list of {"page": int, "text": str}. Returns the same shape, cleaned.
    """
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    counts = Counter()
    for p in pages:
        for line in {norm(l) for l in p["text"].splitlines() if l.strip()}:
            counts[line] += 1
    threshold = max(5, int(0.35 * len(pages))) if len(pages) >= 8 else None

    out = []
    for p in pages:
        kept = []
        for line in p["text"].splitlines():
            s = line.strip()
            if not s or re.fullmatch(r"\d{1,4}", s):
                continue
            if threshold and len(s) <= 120 and counts[norm(s)] >= threshold:
                continue
            kept.append(s)
        text = "\n".join(kept).strip()
        if len(text) >= MIN_PAGE_CHARS:
            out.append({"page": p["page"], "text": text})
    return out


def file_quality(pages):
    """(mean OCR word confidence weighted by words, word count) over a file's OCR'd pages."""
    words = sum(p.get("words", 0) for p in pages.values() if p.get("conf") is not None)
    if not words:
        return None, 0
    total = sum(p["conf"] * p["words"] for p in pages.values() if p.get("conf") is not None)
    return total / words, words


def is_low_quality(pages):
    conf, words = file_quality(pages)
    return conf is not None and words >= 200 and conf < LOW_CONFIDENCE


def ocr_with_confidence(image):
    """OCR one page image -> (text, mean word confidence or None, word count)."""
    import pytesseract

    d = pytesseract.image_to_data(image, lang="eng", output_type=pytesseract.Output.DICT)
    lines, confs = [], []
    cur_key, cur_words = None, []
    for i, word in enumerate(d["text"]):
        w = word.strip()
        if not w:
            continue
        conf = float(d["conf"][i])
        if conf >= 0:
            confs.append(conf)
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if key != cur_key:
            if cur_words:
                lines.append((cur_key, " ".join(cur_words)))
            cur_key, cur_words = key, []
        cur_words.append(w)
    if cur_words:
        lines.append((cur_key, " ".join(cur_words)))

    out, prev = [], None
    for key, line in lines:
        if prev is not None:
            out.append("\n\n" if key[:2] != prev[:2] else "\n")
        out.append(line)
        prev = key
    mean = sum(confs) / len(confs) if confs else None
    return "".join(out).strip(), mean, len(confs)


def process_pages(pdf_path, page_numbers):
    """Worker: read the given 1-based pages of one PDF. Opens the file once per task."""
    import pypdfium2 as pdfium

    results = []
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        for n in page_numbers:
            page = pdf[n - 1]
            try:
                text = (page.get_textpage().get_text_range() or "").strip()
            except Exception:
                text = ""
            if len(text) >= MIN_TEXT_LAYER_CHARS:
                results.append({"page": n, "text": text, "method": "text"})
            else:
                image = page.render(scale=OCR_DPI / 72, grayscale=True).to_pil()
                text, conf, words = ocr_with_confidence(image)
                results.append({"page": n, "text": text, "method": "ocr", "conf": conf, "words": words})
    finally:
        pdf.close()
    return results


def report_quality(name, pages):
    conf, words = file_quality(pages)
    if conf is None:
        return
    if is_low_quality(pages):
        print(f"  WARNING: [{name}] OCR confidence is only {conf:.0f}% over {words} words. "
              "This looks handwritten or poorly scanned, so ingest.py will skip it.")
    else:
        print(f"  OCR confidence {conf:.0f}% over {words} words: good")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--only", help="only PDFs whose file name contains this text")
    ap.add_argument("--max-pages", type=int, help="only the first N pages per file (for testing)")
    args = ap.parse_args()

    if not shutil.which("tesseract"):
        sys.exit("Tesseract is not installed. On macOS run: brew install tesseract")

    pdfs = [p for p in list_pdfs() if not args.only or args.only.lower() in os.path.basename(p).lower()]
    if not pdfs:
        sys.exit(f"No PDFs found in {DOCS_DIR}")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.environ["OMP_THREAD_LIMIT"] = "1"  # one Tesseract thread per worker process

    import pypdfium2 as pdfium

    print(f"{len(pdfs)} PDFs, {args.workers} worker(s)")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for pdf_path in pdfs:
            name = os.path.basename(pdf_path)
            sha = sha256_file(pdf_path)
            doc = pdfium.PdfDocument(pdf_path)
            total = len(doc)
            doc.close()

            out = cache_path(pdf_path)
            meta, done = load_extracted(out)
            if not meta or meta.get("sha256") != sha:
                done = {}
                with open(out, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"file": name, "sha256": sha, "total_pages": total}) + "\n")

            wanted = range(1, min(total, args.max_pages or total) + 1)
            todo = [p for p in wanted if p not in done]
            if not todo:
                print(f"[{name}] already extracted ({len(done)}/{total} pages)")
                report_quality(name, done)
                continue

            print(f"[{name}] {len(done)}/{total} done, extracting {len(todo)} more")
            tasks = [todo[i:i + TASK_PAGES] for i in range(0, len(todo), TASK_PAGES)]
            start, finished = time.time(), 0
            futures = {pool.submit(process_pages, pdf_path, t): t for t in tasks}
            with open(out, "a", encoding="utf-8") as f:
                for fut in as_completed(futures):
                    try:
                        recs = fut.result()
                    except Exception as e:
                        print(f"  task failed ({str(e)[:100]}); re-run to retry those pages")
                        continue
                    for rec in recs:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    finished += len(recs)
                    rate = finished / max(1e-9, time.time() - start)
                    eta = (len(todo) - finished) / rate / 60 if rate else 0
                    print(f"  {len(done) + finished}/{total} pages, about {eta:.0f} min left", flush=True)
            _, done_now = load_extracted(out)
            report_quality(name, done_now)

    print("Extraction finished. Next: python3 ingest.py")


if __name__ == "__main__":
    main()
