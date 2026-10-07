"""Search the extracted text for a phrase. Handy for writing golden_set.json questions.

Usage:  python3 find_pages.py "primitive streak"
Prints the file, page and a snippet for every page that contains the phrase.
Page numbers match the labels used in answers (counted from the first page of the PDF).
"""
import glob
import os
import sys

from extract import EXTRACT_DIR, load_extracted


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    phrase = " ".join(sys.argv[1:]).lower()
    found = 0
    for path in sorted(glob.glob(os.path.join(EXTRACT_DIR, "*.jsonl"))):
        meta, pages = load_extracted(path)
        for n in sorted(pages):
            text = " ".join(pages[n]["text"].split())
            idx = text.lower().find(phrase)
            if idx >= 0:
                found += 1
                snippet = text[max(0, idx - 60): idx + 140]
                print(f"{meta['file']}  p.{n}\n    ...{snippet}...")
    print(f"\n{found} matching page(s)" if found else "No matches. Is extract.py finished?")


if __name__ == "__main__":
    main()
