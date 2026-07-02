"""Build Retrieval-only Top-5 baseline from Dify Hybrid Top-20 candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "setr_reproduction"
    / "results"
    / "candidates"
    / "dify_hybrid_top20_500.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "selections"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def build_selected_candidate(candidate: Dict[str, Any], selection_rank: int, gold_titles: List[str]) -> Dict[str, Any]:
    title = candidate.get("title")
    return {
        "selection_rank": selection_rank,
        "original_rank": candidate.get("rank"),
        "pid": candidate.get("pid"),
        "title": title,
        "text": candidate.get("text"),
        "score": candidate.get("score"),
        "parse_ok": candidate.get("parse_ok"),
        "is_gold_title": title in set(gold_titles) if title else False,
    }


def build_retrieval_top5_rows(candidate_rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for row in candidate_rows:
        gold_titles = row.get("gold_titles", [])
        selected_passages = [
            build_selected_candidate(candidate, idx + 1, gold_titles)
            for idx, candidate in enumerate(row.get("candidates", [])[:top_k])
        ]
        metrics = evidence_metrics(gold_titles, [item.get("title") for item in selected_passages])

        output_rows.append(
            {
                "qid": row.get("qid"),
                "sample_index": row.get("sample_index"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "gold_titles": gold_titles,
                "method": "retrieval_only_top5",
                "candidate_pool": "dify_hybrid_top20_500",
                "selected_k_requested": top_k,
                "selected_count": len(selected_passages),
                "selection_metrics": metrics,
                "top20_metrics": row.get("topk_metrics"),
                "selected_passages": selected_passages,
            }
        )
    return output_rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"samples": 0}

    selected_counts = [row["selected_count"] for row in rows]
    recalls = [row["selection_metrics"]["evidence_recall"] for row in rows]
    precisions = [row["selection_metrics"]["evidence_precision"] for row in rows]
    title_hits = [row["selection_metrics"]["gold_title_hit"] for row in rows]
    all_hits = [row["selection_metrics"]["all_support_hit"] for row in rows]
    parse_failed_selected_count = sum(
        1
        for row in rows
        for item in row["selected_passages"]
        if item.get("parse_ok") is False or not item.get("pid") or not item.get("title")
    )

    selected_k = rows[0]["selected_k_requested"]
    return {
        "samples": total,
        "method": "retrieval_only_top5",
        "candidate_pool": "dify_hybrid_top20_500",
        "selected_k_requested": selected_k,
        "avg_selected_count": sum(selected_counts) / total,
        "min_selected_count": min(selected_counts),
        "max_selected_count": max(selected_counts),
        "not_equal_selected_k_count": sum(1 for count in selected_counts if count != selected_k),
        "empty_selection_count": sum(1 for count in selected_counts if count == 0),
        "parse_failed_selected_count": parse_failed_selected_count,
        "parse_failed_query_count": sum(
            1
            for row in rows
            if any(
                item.get("parse_ok") is False or not item.get("pid") or not item.get("title")
                for item in row["selected_passages"]
            )
        ),
        "selected_evidence_recall": sum(recalls) / total,
        "selected_evidence_precision": sum(precisions) / total,
        "selected_gold_title_hit": sum(title_hits) / total,
        "selected_all_support_hit": sum(all_hits) / total,
    }


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Retrieval-only Top-5 baseline from Dify Hybrid Top-20 candidates."
    )
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES), help="Input candidate JSONL file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top-ranked candidates to select.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidates)
    output_dir = Path(args.output_dir)
    top_k = args.top_k

    output_path = output_dir / "retrieval_top5_hybrid_500.jsonl"
    summary_json_path = output_dir / "retrieval_top5_hybrid_500_summary.json"
    summary_csv_path = output_dir / "retrieval_top5_hybrid_500_summary.csv"

    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")

    if not args.force:
        existing = [path for path in [output_path, summary_json_path, summary_csv_path] if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --force to overwrite: "
                + ", ".join(str(path) for path in existing)
            )

    candidate_rows = read_jsonl(candidate_path)
    output_rows = build_retrieval_top5_rows(candidate_rows, top_k)
    summary = summarize(output_rows)

    write_jsonl(output_path, output_rows)
    write_json(summary_json_path, summary)
    write_summary_csv(summary_csv_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_path)
    print(summary_json_path)
    print(summary_csv_path)


if __name__ == "__main__":
    main()
