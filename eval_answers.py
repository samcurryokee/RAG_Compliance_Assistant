"""End-to-end answer evaluation through the full chain (retrieval + Groq).

Usage:  python3 eval_answers.py [label]
Example: python3 eval_answers.py baseline
         python3 eval_answers.py after-prompt-change

Appends a summary line to eval_history.jsonl so you can compare runs before and after any
change to the prompt, chunking, k or models. One Groq call per question, paced for free tiers
(EVAL_PAUSE_SECONDS, default 3).
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

from chain import ask

PAUSE = float(os.getenv("EVAL_PAUSE_SECONDS", "3"))
# For should_refuse questions, a polite "the notes do not mention this" also counts.
SOFT_REFUSAL_MARKERS = (
    "does not mention",
    "does not contain",
    "does not address",
    "do not mention",
    "no information",
    "not provided",
)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    with open("golden_set.json", encoding="utf-8") as f:
        golden = json.load(f)

    rows = []
    for q in golden:
        try:
            r = ask(q["question"])
        except Exception as e:
            print(f"{q['id']} ERROR: {str(e)[:120]}")
            rows.append({"id": q["id"], "error": True})
            time.sleep(30)
            continue

        cited = r["cited"]
        expected = {(e["file"], p) for e in q["expected"] for p in e["pages"]}
        lowered = r["answer"].lower()
        row = {"id": q["id"], "should_refuse": q["should_refuse"], "error": False}

        if q["should_refuse"]:
            row["pass"] = r["refused"] or any(m in lowered for m in SOFT_REFUSAL_MARKERS)
        else:
            row["false_refusal"] = r["refused"]
            row["bad_citation"] = bool(r["bad_citations"])
            row["no_citation"] = (not r["refused"]) and not cited
            row["cites_expected"] = bool(cited & expected) if expected else None
            row["pass"] = (
                not row["false_refusal"]
                and not row["bad_citation"]
                and not row["no_citation"]
                and row["cites_expected"] is not False
            )

        rows.append(row)
        flags = []
        if q["should_refuse"]:
            flags.append("refused OK" if row["pass"] else "DID NOT REFUSE")
        else:
            if row["false_refusal"]:
                flags.append("FALSE REFUSAL")
            if row["bad_citation"]:
                flags.append(f"BAD CITATIONS {r['bad_citations']}")
            if row["no_citation"]:
                flags.append("NO CITATION")
            if row["cites_expected"] is False:
                flags.append("cited pages do not include the expected ones")
        print(f"{q['id']} [{'PASS' if row['pass'] else 'FAIL'}] {'; '.join(flags)}")
        print(f"     {q['question']}")
        time.sleep(PAUSE)

    ok = [r for r in rows if not r["error"]]
    refuse = [r for r in ok if r["should_refuse"]]
    answer = [r for r in ok if not r["should_refuse"]]
    with_expected = [r for r in answer if r["cites_expected"] is not None]

    summary = {
        "label": label,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": len(golden),
        "errors": len(rows) - len(ok),
        "refusal_tests_passed": sum(r["pass"] for r in refuse),
        "refusal_tests": len(refuse),
        "false_refusals": sum(r["false_refusal"] for r in answer),
        "bad_citation_answers": sum(r["bad_citation"] for r in answer),
        "no_citation_answers": sum(r["no_citation"] for r in answer),
        "cites_expected_page": sum(bool(r["cites_expected"]) for r in with_expected),
        "with_expected_pages": len(with_expected),
        "overall_pass": sum(r["pass"] for r in ok),
        "evaluated": len(ok),
    }

    print("\n--- Answer evaluation ---")
    print(f"Refusal tests passed:        {summary['refusal_tests_passed']}/{summary['refusal_tests']}")
    print(f"False refusals:              {summary['false_refusals']}/{len(answer)}")
    print(f"Answers with bad citations:  {summary['bad_citation_answers']}/{len(answer)}")
    print(f"Answers with no citations:   {summary['no_citation_answers']}/{len(answer)}")
    print(f"Cites an expected page:      {summary['cites_expected_page']}/{summary['with_expected_pages']}")
    print(f"Overall pass:                {summary['overall_pass']}/{summary['evaluated']}")
    if summary["errors"]:
        print(f"Errors (not scored):         {summary['errors']}")

    with open("eval_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print("Saved summary to eval_history.jsonl")


if __name__ == "__main__":
    main()
