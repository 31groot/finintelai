import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from src.agents.graph import run

RESULTS_DIR = Path("eval/results")

NOT_AVAILABLE_MARKERS = [
    "not available in the retrieved context",
    "not available",
    "not found in the context",
    "cannot be found",
    "no information",
    "not provided in the context",
    "not explicitly stated",
    "unable to find",
]


def normalize_number(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[,\s₹$`%]", "", text)
    text = text.strip()
    text = re.sub(r"\.$", "", text)  
    return text


def said_not_available(answer):
    low = (answer or "").lower()
    return any(marker in low for marker in NOT_AVAILABLE_MARKERS)


def extract_numbers(answer):
    """Every number-like token in the answer, normalized."""
    raw = re.findall(r"[-+]?\d[\d,]*\.?\d*", answer or "")
    return {normalize_number(r) for r in raw}


def score_answerable(question, answer):

    answer_type = question.get("answer_type")
    answer_numbers = extract_numbers(answer)

    if answer_type == "range":
        endpoints = [normalize_number(e) for e in question.get("range_endpoints", [])]
        hit = all(e in answer_numbers for e in endpoints)
        return hit, f"range {endpoints} {'all present' if hit else 'missing'}"

    if answer_type == "multi_value":
        expected = [normalize_number(v) for v in question["expected_answer"]]
        present = [v for v in expected if v in answer_numbers]
        hit = len(present) == len(expected)
        return hit, f"{len(present)}/{len(expected)} values present"

    expected = normalize_number(question["expected_answer"])
    hit = expected in answer_numbers
    return hit, f"expected {expected} {'found' if hit else 'MISSING'}"


def score_source(question, retrieved_metadata):
    expected = question.get("expected_source")
    if not expected:
        return None
    expected_list = expected if isinstance(expected, list) else [expected]
    retrieved = {(m or {}).get("source") for m in retrieved_metadata}
    return all(e in retrieved for e in expected_list)


def score_unanswerable(answer):
    return said_not_available(answer)


def run_question(question, companies_hint=None):
    q_companies = []
    comp = question.get("company")
    if comp and comp not in ("multi", None):
        q_companies = [comp]

    result = run(question["question"], companies=q_companies)
    answer = result.get("answer", "")
    retrieved = result.get("metadata", [])

    row = {
        "id": question["id"],
        "category": question["category"],
        "company": question.get("company"),
        "answerable": question["answerable"],
        "retry_count": result.get("retry_count"),
        "grounded": (result.get("verification") or {}).get("grounded"),
        "answer": answer,
    }

    if question["answerable"]:
        hit, detail = score_answerable(question, answer)
        row["correct"] = hit
        row["detail"] = detail
        src = score_source(question, retrieved)
        row["source_hit"] = src
    else:
        declined = score_unanswerable(answer)
        row["correct"] = declined
        row["hallucinated"] = not declined
        row["detail"] = "declined" if declined else "FABRICATED an answer"
        row["source_hit"] = None

    return row


def summarize(rows):
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    correct = sum(1 for r in answerable if r["correct"])
    print(f"\nExact-match accuracy:  {correct}/{len(answerable)}  "
          f"({100 * correct / len(answerable):.0f}%)" if answerable else "no answerable questions")

    src_scored = [r for r in answerable if r["source_hit"] is not None]
    src_hit = sum(1 for r in src_scored if r["source_hit"])
    if src_scored:
        print(f"Source retrieved:      {src_hit}/{len(src_scored)}  "
              f"({100 * src_hit / len(src_scored):.0f}%)")

    if unanswerable:
        halluc = sum(1 for r in unanswerable if r.get("hallucinated"))
        print(f"Hallucination rate:    {halluc}/{len(unanswerable)}  "
              f"({100 * halluc / len(unanswerable):.0f}% of unanswerable got a fabricated answer)")

    retries = [r["retry_count"] for r in rows if r["retry_count"] is not None]
    if retries:
        retried = sum(1 for c in retries if c > 1)
        print(f"Retry fired:           {retried}/{len(retries)} questions")

    grounded_rows = [r for r in rows if r.get("grounded") is not None]
    if grounded_rows:
        grounded = sum(1 for r in grounded_rows if r["grounded"])
        print(f"Grounded (verifier):   {grounded}/{len(grounded_rows)}  "
              f"({100 * grounded / len(grounded_rows):.0f}%)")

    both = [r for r in rows if r["answerable"] and r.get("grounded") is not None]
    if both:
        correct_and_grounded = sum(1 for r in both if r["correct"] and r["grounded"])
        correct_not_grounded = sum(1 for r in both if r["correct"] and not r["grounded"])
        wrong_but_grounded = sum(1 for r in both if not r["correct"] and r["grounded"])
        print(f"  correct & grounded:   {correct_and_grounded}")
        print(f"  correct, not grounded:{correct_not_grounded}  (verifier too strict?)")
        print(f"  wrong but grounded:   {wrong_but_grounded}  (verifier missed it)")

    print("\nBy category:")
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        by_cat[r["category"]][0] += int(bool(r["correct"]))
        by_cat[r["category"]][1] += 1
    for cat in sorted(by_cat):
        c, n = by_cat[cat]
        print(f"  {cat:<20} {c}/{n}  ({100 * c / n:.0f}%)")

    print("\nBy company:")
    by_co = defaultdict(lambda: [0, 0])
    for r in rows:
        by_co[str(r["company"])][0] += int(bool(r["correct"]))
        by_co[str(r["company"])][1] += 1
    for co in sorted(by_co):
        c, n = by_co[co]
        print(f"  {co:<20} {c}/{n}  ({100 * c / n:.0f}%)")

    print("\nFailures:")
    for r in rows:
        if not r["correct"]:
            src = "" if r["source_hit"] is None else (
                "  [source retrieved]" if r["source_hit"] else "  [SOURCE MISSED]"
            )
            print(f"  {r['id']:<14} {r['detail']}{src}")


def main(path, delay=1.0):
    questions = json.loads(Path(path).read_text())
    print(f"Running {len(questions)} questions...\n")

    rows = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q['id']}", flush=True)
        try:
            rows.append(run_question(q))
        except Exception as e:
            print(f"      ERROR: {type(e).__name__}: {e}")
            rows.append({
                "id": q["id"], "category": q["category"],
                "company": q.get("company"), "answerable": q["answerable"],
                "correct": False, "detail": f"ERROR {type(e).__name__}",
                "source_hit": None, "retry_count": None, "grounded": None,
                "answer": "",
            })
        time.sleep(delay)

    summarize(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"run_{stamp}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nFull results written to {out}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "src/eval/eval_questions.json"
    main(path,)