"""Retrieval evaluation: does a chunk from the expected file and page show up in the top-k?

Usage:  python3 eval_retrieval.py
Reads golden_set.json (refusal questions and entries without expected pages are skipped).
No LLM calls, only one embedding call per question.
"""
import json

from chain import get_store, page_span, source_label

RETRIEVE_K = 10  # fetch 10 once, then score at k=3, k=5 and MRR


def is_hit(doc, expected):
    start, end = page_span(doc)
    name = source_label(doc)
    return any(
        e["file"] == name and any(start <= p <= end for p in e["pages"]) for e in expected
    )


def main():
    with open("golden_set.json", encoding="utf-8") as f:
        golden = json.load(f)

    testable = [q for q in golden if not q["should_refuse"] and q["expected"]]
    if not testable:
        print("Nothing to evaluate. Add expected pages to golden_set.json first.")
        return

    store = get_store()
    hits3 = hits5 = 0
    rr_total = 0.0
    misses = []

    for q in testable:
        docs = store.similarity_search(q["question"], k=RETRIEVE_K)
        rank = next((i for i, d in enumerate(docs, start=1) if is_hit(d, q["expected"])), None)
        if rank is not None and rank <= 3:
            hits3 += 1
        if rank is not None and rank <= 5:
            hits5 += 1
        if rank is not None:
            rr_total += 1.0 / rank
        else:
            misses.append(q["id"])
        got = [f"{source_label(d)[:18]} p.{page_span(d)[0]}-{page_span(d)[1]}" for d in docs[:3]]
        print(f"{q['id']} [{'rank ' + str(rank) if rank else 'MISS'}] {q['question']}")
        print(f"     expected {[(e['file'][:18], e['pages']) for e in q['expected']]} | top 3: {got}")

    n = len(testable)
    print("\n--- Retrieval results ---")
    print(f"Questions evaluated: {n}")
    print(f"Hit rate @3: {hits3}/{n} = {hits3 / n:.0%}")
    print(f"Hit rate @5: {hits5}/{n} = {hits5 / n:.0%}")
    print(f"MRR (up to k={RETRIEVE_K}): {rr_total / n:.3f}")
    if misses:
        print(f"Misses (not in top {RETRIEVE_K}): {', '.join(misses)}")


if __name__ == "__main__":
    main()
