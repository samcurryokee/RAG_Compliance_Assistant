"""Helper for building golden_set.json: find which PDF page an article starts on.

Usage:
  python3 find_pages.py 137 145 142        # look up articles by number
  python3 find_pages.py --text "phrase"    # search for a phrase
  python3 find_pages.py --golden           # every article_hint in golden_set.json

Page numbers count from the first page of the PDF, matching chain.py.
Matches far into the document (counties, schedules) are usually false positives.
"""
import glob
import json
import os
import re
import sys

from pypdf import PdfReader

DOCS_DIR = "data/documents"


def load_pages():
    pages = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf"))):
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            pages.append((os.path.basename(path), i + 1, page.extract_text() or ""))
    return pages


def looks_like_toc(line):
    return bool(re.search(r"\.{3,}", line) or re.search(r"\s\d{1,3}\s*$", line))


def article_report(n, pages):
    heading_re = re.compile(rf"^\s*{n}\.\s+(.+)$", re.M)
    mention_re = re.compile(rf"\bArticles?\s+{n}\b")
    starts, tocs, mentions = [], [], []
    for _, num, text in pages:
        for m in heading_re.finditer(text):
            line = m.group(0).strip()
            (tocs if looks_like_toc(line) else starts).append((num, line[:80]))
        if mention_re.search(text):
            mentions.append(num)

    print(f"\nArticle {n}")
    if starts:
        for num, line in starts:
            print(f"  starts on p.{num}: {line}")
        print("  (long articles run onto the next page, so check p.N+1 too)")
    else:
        print("  no heading found; try --text with a phrase from the article")
    if tocs:
        print(f"  likely table of contents: pages {sorted({p for p, _ in tocs})}")
    if mentions:
        print(f"  mentioned in text on pages: {sorted(set(mentions))}")


def text_report(phrase, pages):
    print(f'\nPages containing "{phrase}":')
    found = False
    for _, num, text in pages:
        if phrase.lower() in text.lower():
            found = True
            idx = text.lower().index(phrase.lower())
            snippet = text[max(0, idx - 40) : idx + 80].replace("\n", " ")
            print(f"  p.{num}: ...{snippet}...")
    if not found:
        print("  no matches")


def golden_report(pages):
    with open("golden_set.json") as f:
        golden = json.load(f)
    for q in golden:
        m = re.search(r"Article\s+(\d+)", q.get("article_hint", ""))
        print(f"\n{q['id']}: {q['question']}")
        if m:
            article_report(int(m.group(1)), pages)
        else:
            print("  no article number in hint, skipping")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    pages = load_pages()
    print(f"Loaded {len(pages)} pages")
    if args[0] == "--golden":
        golden_report(pages)
    elif args[0] == "--text":
        text_report(" ".join(args[1:]), pages)
    else:
        for a in args:
            if a.isdigit():
                article_report(int(a), pages)


if __name__ == "__main__":
    main()
