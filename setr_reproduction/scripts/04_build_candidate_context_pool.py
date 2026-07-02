import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def evidence_metrics(gold_titles: List[str], selected_titles: List[Optional[str]]) -> Dict[str, Any]:
    gold_set = set(gold_titles)
    selected = [title for title in selected_titles if title]
    selected_set = set(selected)
    hit_count = len(gold_set & selected_set)
    return {
        "evidence_recall": hit_count / len(gold_set) if gold_set else 0.0,
        "evidence_precision": hit_count / len(selected) if selected else 0.0,
        "gold_title_hit": int(hit_count > 0),
        "all_support_hit": int(gold_set.issubset(selected_set)) if gold_set else 0,
        "hit_gold_titles": sorted(gold_set & selected_set),
        "missing_gold_titles": sorted(gold_set - selected_set),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"samples": 0}

    returned_counts = [row["actual_returned_count"] for row in rows]
    recalls = [row["topk_metrics"]["evidence_recall"] for row in rows]
    precisions = [row["topk_metrics"]["evidence_precision"] for row in rows]
    title_hits = [row["topk_metrics"]["gold_title_hit"] for row in rows]
    all_hits = [row["topk_metrics"]["all_support_hit"] for row in rows]
    parse_failed_counts = [row["parse_failed_count"] for row in rows]
    return {
        "samples": total,
        "avg_returned_count": sum(returned_counts) / total,
        "min_returned_count": min(returned_counts),
        "max_returned_count": max(returned_counts),
        "less_than_top_k_count": sum(1 for row in rows if row["actual_returned_count"] < row["requested_top_k"]),
        "empty_retrieval_count": sum(1 for count in returned_counts if count == 0),
        "parse_failed_candidate_count": sum(parse_failed_counts),
        "parse_failed_query_count": sum(1 for count in parse_failed_counts if count > 0),
        "topk_evidence_recall": sum(recalls) / total,
        "topk_evidence_precision": sum(precisions) / total,
        "topk_gold_title_hit": sum(title_hits) / total,
        "topk_all_support_hit": sum(all_hits) / total,
    }


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified candidate pool from HotpotQA provided candidate contexts.")
    parser.add_argument(
        "--input",
        default="setr_reproduction/data/processed/eval/candidate_context_500.jsonl",
    )
    parser.add_argument(
        "--output",
        default="setr_reproduction/results/candidates/candidate_context_pool_500.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        default="setr_reproduction/results/candidates/candidate_context_pool_500_summary.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="setr_reproduction/results/candidates/candidate_context_pool_500_summary.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_json_path = Path(args.summary_json)
    summary_csv_path = Path(args.summary_csv)

    rows = []
    for row in read_jsonl(input_path):
        candidates = []
        for rank, item in enumerate(row["candidate_contexts"], start=1):
            candidates.append(
                {
                    "rank": rank,
                    "pid": item["local_pid"],
                    "title": item["title"],
                    "text": item["text"],
                    "raw_content": item["content"],
                    "score": None,
                    "segment_id": None,
                    "document_id": None,
                    "document_name": "hotpotqa_candidate_context",
                    "position": item["context_index"],
                    "tokens": None,
                    "word_count": len(item["text"].split()),
                    "parse_ok": True,
                    "is_gold_title": item["is_gold_title"],
                }
            )

        metrics = evidence_metrics(row["gold_titles"], [item["title"] for item in candidates])
        rows.append(
            {
                "qid": row["qid"],
                "sample_index": row["sample_index"],
                "question": row["question"],
                "answer": row["answer"],
                "gold_titles": row["gold_titles"],
                "retrieval_backend": "hotpotqa_candidate_context",
                "retrieval_label": "candidate_context",
                "requested_top_k": 20,
                "actual_returned_count": len(candidates),
                "parse_failed_count": 0,
                "latency_seconds": 0.0,
                "topk_metrics": metrics,
                "candidates": candidates,
            }
        )

    write_jsonl(output_path, rows)
    summary = summarize(rows)
    write_json(summary_json_path, summary)
    write_summary_csv(summary_csv_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_path)
    print(summary_csv_path)


if __name__ == "__main__":
    main()
