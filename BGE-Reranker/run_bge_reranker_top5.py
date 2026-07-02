"""Run BGE-Reranker Top-5 baseline on Dify Hybrid Top-20 candidates.

This is a minimal local implementation for SETR reproduction step 4.
It intentionally avoids service wrappers and only writes experiment artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


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


def detect_fp16(default: bool) -> bool:
    if not default:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_reranker(model_name_or_path: str, use_fp16: bool) -> Any:
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RuntimeError(
            "FlagEmbedding is not installed. Run: pip install -r BGE-Reranker/requirements.txt"
        ) from exc

    return FlagReranker(model_name_or_path, use_fp16=use_fp16)


def normalize_scores(scores: Any) -> List[float]:
    if isinstance(scores, (int, float)):
        return [float(scores)]
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return [float(score) for score in scores]


def compute_scores(
    reranker: Any,
    pairs: Sequence[Sequence[str]],
    batch_size: int,
    normalize: bool,
) -> List[float]:
    """Call FlagReranker with compatibility for slightly different versions."""

    try:
        scores = reranker.compute_score(pairs, batch_size=batch_size, normalize=normalize)
    except TypeError:
        try:
            scores = reranker.compute_score(pairs, batch_size=batch_size)
        except TypeError:
            scores = reranker.compute_score(pairs)
    return normalize_scores(scores)


def candidate_text(candidate: Dict[str, Any]) -> str:
    title = candidate.get("title") or ""
    text = candidate.get("text") or ""
    if title and text:
        return f"{title}. {text}"
    return text or title


def rerank_one_row(
    row: Dict[str, Any],
    reranker: Any,
    top_k: int,
    batch_size: int,
    normalize_scores_flag: bool,
) -> Dict[str, Any]:
    question = row.get("question") or ""
    gold_titles = row.get("gold_titles", [])
    candidates = row.get("candidates", [])
    pairs = [[question, candidate_text(candidate)] for candidate in candidates]

    start = time.time()
    scores = compute_scores(reranker, pairs, batch_size=batch_size, normalize=normalize_scores_flag) if pairs else []
    latency = time.time() - start

    scored_candidates: List[Dict[str, Any]] = []
    for candidate, bge_score in zip(candidates, scores):
        enriched = dict(candidate)
        enriched["bge_score"] = bge_score
        scored_candidates.append(enriched)

    scored_candidates.sort(key=lambda item: item.get("bge_score", float("-inf")), reverse=True)

    selected_passages = []
    gold_title_set = set(gold_titles)
    for selection_rank, candidate in enumerate(scored_candidates[:top_k], start=1):
        title = candidate.get("title")
        selected_passages.append(
            {
                "selection_rank": selection_rank,
                "original_rank": candidate.get("rank"),
                "pid": candidate.get("pid"),
                "title": title,
                "text": candidate.get("text"),
                "retrieval_score": candidate.get("score"),
                "bge_score": candidate.get("bge_score"),
                "parse_ok": candidate.get("parse_ok"),
                "is_gold_title": title in gold_title_set if title else False,
            }
        )

    metrics = evidence_metrics(gold_titles, [item.get("title") for item in selected_passages])

    return {
        "qid": row.get("qid"),
        "sample_index": row.get("sample_index"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "gold_titles": gold_titles,
        "method": "bge_reranker_top5",
        "candidate_pool": "dify_hybrid_top20_500",
        "selected_k_requested": top_k,
        "selected_count": len(selected_passages),
        "selection_latency_seconds": latency,
        "selection_metrics": metrics,
        "top20_metrics": row.get("topk_metrics"),
        "selected_passages": selected_passages,
    }


def summarize(rows: List[Dict[str, Any]], model_name_or_path: str, use_fp16: bool) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"samples": 0}

    selected_counts = [row["selected_count"] for row in rows]
    recalls = [row["selection_metrics"]["evidence_recall"] for row in rows]
    precisions = [row["selection_metrics"]["evidence_precision"] for row in rows]
    title_hits = [row["selection_metrics"]["gold_title_hit"] for row in rows]
    all_hits = [row["selection_metrics"]["all_support_hit"] for row in rows]
    latencies = [row["selection_latency_seconds"] for row in rows]
    parse_failed_selected_count = sum(
        1
        for row in rows
        for item in row["selected_passages"]
        if item.get("parse_ok") is False or not item.get("pid") or not item.get("title")
    )

    selected_k = rows[0]["selected_k_requested"]
    return {
        "samples": total,
        "method": "bge_reranker_top5",
        "candidate_pool": "dify_hybrid_top20_500",
        "model_name_or_path": model_name_or_path,
        "use_fp16": use_fp16,
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
        "avg_selection_latency_seconds": sum(latencies) / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BGE-Reranker Top-5 baseline.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES), help="Input Dify Hybrid Top-20 JSONL.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Selection output directory.")
    parser.add_argument("--model-name-or-path", default="BAAI/bge-reranker-base", help="Hugging Face model id or local path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of passages selected after reranking.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size passed to BGE-Reranker.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N samples. 0 means all.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 even when CUDA is available.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalized BGE scores if supported.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidates)
    output_dir = Path(args.output_dir)

    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")

    output_path = output_dir / "bge_reranker_top5_hybrid_500.jsonl"
    summary_json_path = output_dir / "bge_reranker_top5_hybrid_500_summary.json"
    summary_csv_path = output_dir / "bge_reranker_top5_hybrid_500_summary.csv"

    if not args.force:
        existing = [path for path in [output_path, summary_json_path, summary_csv_path] if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --force to overwrite: "
                + ", ".join(str(path) for path in existing)
            )

    use_fp16 = detect_fp16(default=not args.no_fp16)
    print(f"Loading reranker: {args.model_name_or_path}, use_fp16={use_fp16}", flush=True)
    reranker = load_reranker(args.model_name_or_path, use_fp16=use_fp16)

    candidate_rows = read_jsonl(candidate_path)
    if args.limit > 0:
        candidate_rows = candidate_rows[: args.limit]

    output_rows: List[Dict[str, Any]] = []
    total_start = time.time()
    for index, row in enumerate(candidate_rows, start=1):
        output_rows.append(
            rerank_one_row(
                row=row,
                reranker=reranker,
                top_k=args.top_k,
                batch_size=args.batch_size,
                normalize_scores_flag=not args.no_normalize,
            )
        )
        if index % 10 == 0 or index == len(candidate_rows):
            elapsed = time.time() - total_start
            print(f"processed={index}/{len(candidate_rows)} elapsed={elapsed:.1f}s", flush=True)

    summary = summarize(output_rows, args.model_name_or_path, use_fp16)
    write_jsonl(output_path, output_rows)
    write_json(summary_json_path, summary)
    write_summary_csv(summary_csv_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_path)
    print(summary_json_path)
    print(summary_csv_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
