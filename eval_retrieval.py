import json
import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

COLLECTION_NAME = "kenyan_constitution"
RETRIEVE_K = 10  # fetch 10 once, then score at k=3, k=5, and MRR

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=os.environ["DATABASE_URL"],
    use_jsonb=True,
)


def page_label(doc):
    return int(doc.metadata.get("page", 0)) + 1


def first_hit_rank(retrieved_pages, expected_pages):
    """1-based rank of the first chunk from any expected page, or None."""
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            return rank
    return None


def main():
    with open("golden_set.json") as f:
        golden = json.load(f)

    testable = [q for q in golden if not q["should_refuse"] and q["expected_pages"]]
    skipped = [q["id"] for q in golden if not q["should_refuse"] and not q["expected_pages"]]

    if skipped:
        print(f"Skipping {len(skipped)} questions with no expected_pages yet: {', '.join(skipped)}\n")
    if not testable:
        print("Nothing to evaluate. Fill in expected_pages in golden_set.json first.")
        return

    hits3 = hits5 = 0
    rr_total = 0.0
    misses = []

    for q in testable:
        docs = store.similarity_search(q["question"], k=RETRIEVE_K)
        pages = [page_label(d) for d in docs]
        rank = first_hit_rank(pages, q["expected_pages"])

        if rank is not None and rank <= 3:
            hits3 += 1
        if rank is not None and rank <= 5:
            hits5 += 1
        if rank is not None:
            rr_total += 1.0 / rank
        else:
            misses.append(q["id"])

        status = f"rank {rank}" if rank else "MISS"
        print(f"{q['id']} [{status}] expected {q['expected_pages']} | retrieved {pages[:5]}")
        print(f"     {q['question']}")

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