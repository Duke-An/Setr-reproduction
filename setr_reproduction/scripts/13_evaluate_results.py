"""Evaluate SETR reproduction results for step 10.

This script is offline-only. It reads selection files and answer generation
files, then writes the main result tables required by the reproduction plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "setr_reproduction" / "data" / "processed" / "eval"
SELECTION_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "selections"
GENERATION_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "generations"
TABLE_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "tables"


METHODS = {
    "retrieval_top5": {
        "label": "Retrieval Top-5",
        "selection_file": "retrieval_top5_hybrid_500.jsonl",
        "generation_file": "retrieval_top5_hybrid_500_answers.jsonl",
        "selection_summary_file": "retrieval_top5_hybrid_500_summary.json",
        "method_type": "top5",
    },
    "bge_reranker_top5": {
        "label": "BGE-Reranker Top-5",
        "selection_file": "bge_reranker_top5_hybrid_500.jsonl",
        "generation_file": "bge_reranker_top5_hybrid_500_answers.jsonl",
        "selection_summary_file": "bge_reranker_top5_hybrid_500_summary.json",
        "method_type": "top5",
    },
    "llm_listwise_top5": {
        "label": "LLM Listwise Top-5",
        "selection_file": "llm_listwise_top5_hybrid_500.jsonl",
        "generation_file": "llm_listwise_top5_hybrid_500_answers.jsonl",
        "selection_summary_file": "llm_listwise_top5_hybrid_500_summary.json",
        "method_type": "top5",
    },
    "setr_selection_only": {
        "label": "SetR-Selection only",
        "selection_file": "setr_selection_only_hybrid_500.jsonl",
        "generation_file": "setr_selection_only_hybrid_500_answers.jsonl",
        "selection_summary_file": "setr_selection_only_hybrid_500_summary.json",
        "method_type": "setr",
    },
    "setr_cot": {
        "label": "SetR-CoT",
        "selection_file": "setr_cot_hybrid_500.jsonl",
        "generation_file": "setr_cot_hybrid_500_answers.jsonl",
        "selection_summary_file": "setr_cot_hybrid_500_summary.json",
        "method_type": "setr",
    },
    "setr_cot_iri": {
        "label": "SetR-CoT + IRI",
        "selection_file": "setr_cot_iri_hybrid_500.jsonl",
        "generation_file": "setr_cot_iri_hybrid_500_answers.jsonl",
        "selection_summary_file": "setr_cot_iri_hybrid_500_summary.json",
        "method_type": "setr",
    },
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_answer(text: Any) -> str:
    """Normalize answers using the standard QA EM/F1 convention."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def exact_match(prediction: Any, gold: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_contains(prediction: Any, gold: Any) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    if not normalized_gold:
        return 0.0
    return float(normalized_gold in normalized_prediction)


def token_f1(prediction: Any, gold: Any) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def safe_mean(values: Iterable[Optional[float]]) -> float:
    numeric_values = [float(value) for value in values if value is not None]
    return mean(numeric_values) if numeric_values else 0.0


def round_float(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def get_qid(row: Dict[str, Any]) -> str:
    qid = row.get("qid") or row.get("_id") or row.get("id")
    if qid is None:
        raise ValueError(f"Row has no qid/_id/id field: {row}")
    return str(qid)


def selected_count(row: Dict[str, Any]) -> int:
    if row.get("selected_count") is not None:
        return int(row["selected_count"])
    return len(row.get("selected_passages", []))


def get_selection_metric(row: Dict[str, Any], key: str) -> Optional[float]:
    metrics = row.get("selection_metrics") or {}
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def validate_probability(value: float, metric_name: str, method: str) -> None:
    if value < -1e-9 or value > 1 + 1e-9:
        raise ValueError(f"{metric_name} out of range for {method}: {value}")


def load_query_map(query_path: Path, limit: int) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(query_path)
    if limit > 0:
        rows = rows[:limit]
    return {get_qid(row): row for row in rows}


def index_by_qid(rows: List[Dict[str, Any]], source_name: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = get_qid(row)
        if qid in indexed:
            raise ValueError(f"Duplicate qid in {source_name}: {qid}")
        indexed[qid] = row
    return indexed


def parse_methods(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(METHODS.keys())
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {list(METHODS)}")
    return methods


def build_per_sample_rows(
    method: str,
    method_config: Dict[str, Any],
    query_map: Dict[str, Dict[str, Any]],
    selection_rows: List[Dict[str, Any]],
    generation_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    selection_by_qid = index_by_qid(selection_rows, f"{method} selection")
    generation_by_qid = index_by_qid(generation_rows, f"{method} generation")

    missing_generations = sorted(set(selection_by_qid) - set(generation_by_qid))
    extra_generations = sorted(set(generation_by_qid) - set(selection_by_qid))
    if missing_generations or extra_generations:
        raise ValueError(
            f"QID mismatch for {method}: "
            f"missing_generations={len(missing_generations)}, extra_generations={len(extra_generations)}"
        )

    rows: List[Dict[str, Any]] = []
    for qid, selection_row in selection_by_qid.items():
        generation_row = generation_by_qid[qid]
        query_row = query_map.get(qid, {})
        gold_answer = (
            generation_row.get("gold_answer")
            or selection_row.get("answer")
            or query_row.get("answer")
            or ""
        )
        generated_answer = generation_row.get("generated_answer") or ""

        row = {
            "qid": qid,
            "sample_index": selection_row.get("sample_index"),
            "method": method,
            "method_label": method_config["label"],
            "question": selection_row.get("question") or query_row.get("question"),
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
            "answer_em": exact_match(generated_answer, gold_answer),
            "answer_f1": token_f1(generated_answer, gold_answer),
            "answer_contains": answer_contains(generated_answer, gold_answer),
            "gold_title_hit": get_selection_metric(selection_row, "gold_title_hit"),
            "evidence_recall": get_selection_metric(selection_row, "evidence_recall"),
            "evidence_precision": get_selection_metric(selection_row, "evidence_precision"),
            "all_support_hit": get_selection_metric(selection_row, "all_support_hit"),
            "selected_count": selected_count(selection_row),
            "context_word_count": int(generation_row.get("context_word_count") or 0),
            "context_char_count": int(generation_row.get("context_char_count") or 0),
            "generation_latency_seconds": float(generation_row.get("generation_latency_seconds") or 0),
            "generation_parse_success": bool(generation_row.get("parse_success")),
            "generation_failed": bool(generation_row.get("generation_failed")),
        }
        rows.append(row)

    rows.sort(key=lambda item: (item.get("sample_index") is None, item.get("sample_index") or 0, item["qid"]))
    return rows


def summarize_method(
    method: str,
    method_config: Dict[str, Any],
    per_sample_rows: List[Dict[str, Any]],
    selection_summary: Dict[str, Any],
    generation_summary: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    samples = len(per_sample_rows)
    if samples == 0:
        raise ValueError(f"No rows to summarize for {method}")

    answer_em_value = safe_mean(row["answer_em"] for row in per_sample_rows)
    answer_f1_value = safe_mean(row["answer_f1"] for row in per_sample_rows)
    answer_contains_value = safe_mean(row["answer_contains"] for row in per_sample_rows)
    gold_title_hit_value = safe_mean(row["gold_title_hit"] for row in per_sample_rows)
    evidence_recall_value = safe_mean(row["evidence_recall"] for row in per_sample_rows)
    evidence_precision_value = safe_mean(row["evidence_precision"] for row in per_sample_rows)
    all_support_hit_value = safe_mean(row["all_support_hit"] for row in per_sample_rows)
    avg_selected = safe_mean(row["selected_count"] for row in per_sample_rows)
    avg_context_words = safe_mean(row["context_word_count"] for row in per_sample_rows)
    avg_context_chars = safe_mean(row["context_char_count"] for row in per_sample_rows)
    avg_generation_latency = safe_mean(row["generation_latency_seconds"] for row in per_sample_rows)

    selection_parse_failure_rate = 1.0 - float(selection_summary.get("parse_success_rate", 1.0))
    selection_fallback_rate = float(selection_summary.get("fallback_rate", 0.0))
    generation_parse_failure_rate = 1.0 - safe_mean(float(row["generation_parse_success"]) for row in per_sample_rows)
    generation_failure_rate = safe_mean(float(row["generation_failed"]) for row in per_sample_rows)
    avg_selection_latency = float(selection_summary.get("avg_selection_latency_seconds", 0.0) or 0.0)
    avg_total_latency = avg_selection_latency + avg_generation_latency

    for metric_name, value in {
        "answer_em": answer_em_value,
        "answer_f1": answer_f1_value,
        "answer_contains": answer_contains_value,
        "gold_title_hit": gold_title_hit_value,
        "evidence_recall": evidence_recall_value,
        "evidence_precision": evidence_precision_value,
        "all_support_hit": all_support_hit_value,
        "selection_parse_failure_rate": selection_parse_failure_rate,
        "selection_fallback_rate": selection_fallback_rate,
        "generation_parse_failure_rate": generation_parse_failure_rate,
        "generation_failure_rate": generation_failure_rate,
    }.items():
        validate_probability(value, metric_name, method)

    if method_config["method_type"] == "top5" and abs(avg_selected - 5.0) > 1e-9:
        raise ValueError(f"Top-5 method should have avg_selected_passages=5 for {method}, got {avg_selected}")
    if method_config["method_type"] == "setr" and avg_selected > 5.0 + 1e-9:
        raise ValueError(f"SetR method should have avg_selected_passages <= 5 for {method}, got {avg_selected}")

    main_row = {
        "method": method,
        "method_label": method_config["label"],
        "samples": samples,
        "answer_em": answer_em_value,
        "answer_f1": answer_f1_value,
        "answer_contains": answer_contains_value,
        "gold_title_hit": gold_title_hit_value,
        "evidence_recall": evidence_recall_value,
        "evidence_precision": evidence_precision_value,
        "all_support_hit": all_support_hit_value,
        "avg_selected_passages": avg_selected,
        "avg_context_words": avg_context_words,
        "avg_context_chars": avg_context_chars,
        "selection_parse_failure_rate": selection_parse_failure_rate,
        "selection_fallback_rate": selection_fallback_rate,
        "generation_parse_failure_rate": generation_parse_failure_rate,
        "generation_failure_rate": generation_failure_rate,
        "avg_selection_latency_seconds": avg_selection_latency,
        "avg_generation_latency_seconds": avg_generation_latency,
        "avg_total_latency_seconds": avg_total_latency,
    }

    evidence_row = {
        "method": method,
        "method_label": method_config["label"],
        "samples": samples,
        "gold_title_hit": gold_title_hit_value,
        "evidence_recall": evidence_recall_value,
        "evidence_precision": evidence_precision_value,
        "all_support_hit": all_support_hit_value,
        "avg_selected_passages": avg_selected,
    }

    efficiency_row = {
        "method": method,
        "method_label": method_config["label"],
        "samples": samples,
        "avg_selected_passages": avg_selected,
        "avg_context_words": avg_context_words,
        "avg_context_chars": avg_context_chars,
        "selection_parse_success_rate": float(selection_summary.get("parse_success_rate", 1.0)),
        "selection_fallback_rate": selection_fallback_rate,
        "generation_parse_success_rate": float(generation_summary.get("parse_success_rate", 0.0) or 0.0),
        "generation_failure_rate": generation_failure_rate,
        "avg_selection_latency_seconds": avg_selection_latency,
        "avg_generation_latency_seconds": avg_generation_latency,
        "avg_total_latency_seconds": avg_total_latency,
    }

    return (
        {key: round_float(value) for key, value in main_row.items()},
        {key: round_float(value) for key, value in evidence_row.items()},
        {key: round_float(value) for key, value in efficiency_row.items()},
    )


def output_paths(suffix: str) -> Dict[str, Path]:
    return {
        "main": TABLE_DIR / f"main_results{suffix}.csv",
        "evidence": TABLE_DIR / f"evidence_results{suffix}.csv",
        "efficiency": TABLE_DIR / f"efficiency_results{suffix}.csv",
        "per_sample": TABLE_DIR / f"per_sample_evaluation{suffix}.jsonl",
        "summary": TABLE_DIR / f"evaluation_summary{suffix}.json",
    }


def ensure_outputs_can_be_written(paths: Dict[str, Path], force: bool) -> None:
    if force:
        return
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Output files already exist. Use --force to overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SETR reproduction step 10 results.")
    parser.add_argument("--methods", default="all", help="Comma-separated methods or 'all'.")
    parser.add_argument("--queries", default=str(DATA_DIR / "queries_500.jsonl"))
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate first N rows only. 0 means all.")
    parser.add_argument(
        "--allow-missing-generations",
        action="store_true",
        help="Skip methods whose generation files do not exist. Useful before step 9 fully finishes.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow non-500 rows in full run. Use only for debugging.",
    )
    parser.add_argument("--output-suffix", default="", help="Suffix before file extension, e.g. '_debug'.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    query_path = Path(args.queries)
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    paths = output_paths(args.output_suffix)
    ensure_outputs_can_be_written(paths, force=args.force)

    query_map = load_query_map(query_path, limit=args.limit)
    main_rows: List[Dict[str, Any]] = []
    evidence_rows: List[Dict[str, Any]] = []
    efficiency_rows: List[Dict[str, Any]] = []
    all_per_sample_rows: List[Dict[str, Any]] = []
    skipped_methods: List[Dict[str, str]] = []

    for method in methods:
        method_config = METHODS[method]
        selection_path = SELECTION_DIR / method_config["selection_file"]
        generation_path = GENERATION_DIR / method_config["generation_file"]
        selection_summary_path = SELECTION_DIR / method_config["selection_summary_file"]
        generation_summary_path = GENERATION_DIR / method_config["generation_file"].replace(
            "_answers.jsonl",
            "_generation_summary.json",
        )

        if not selection_path.exists():
            raise FileNotFoundError(f"Selection file not found for {method}: {selection_path}")
        if not generation_path.exists():
            if args.allow_missing_generations:
                skipped_methods.append({"method": method, "reason": f"missing generation file: {generation_path}"})
                continue
            raise FileNotFoundError(
                f"Generation file not found for {method}: {generation_path}. "
                "Run step 9 first or use --allow-missing-generations for partial evaluation."
            )

        selection_rows = read_jsonl(selection_path)
        generation_rows = read_jsonl(generation_path)
        if args.limit > 0:
            selection_rows = selection_rows[: args.limit]
            generation_rows = generation_rows[: args.limit]

        if args.limit == 0 and not args.allow_incomplete:
            if len(selection_rows) != args.expected_samples:
                raise ValueError(f"{method} selection rows={len(selection_rows)}, expected={args.expected_samples}")
            if len(generation_rows) != args.expected_samples:
                raise ValueError(f"{method} generation rows={len(generation_rows)}, expected={args.expected_samples}")

        selection_summary = read_json_if_exists(selection_summary_path)
        generation_summary = read_json_if_exists(generation_summary_path)
        per_sample_rows = build_per_sample_rows(
            method=method,
            method_config=method_config,
            query_map=query_map,
            selection_rows=selection_rows,
            generation_rows=generation_rows,
        )
        main_row, evidence_row, efficiency_row = summarize_method(
            method=method,
            method_config=method_config,
            per_sample_rows=per_sample_rows,
            selection_summary=selection_summary,
            generation_summary=generation_summary,
        )

        main_rows.append(main_row)
        evidence_rows.append(evidence_row)
        efficiency_rows.append(efficiency_row)
        all_per_sample_rows.extend(per_sample_rows)

    if not main_rows:
        raise ValueError("No methods were evaluated. Check generation files and method selection.")

    write_csv(paths["main"], main_rows)
    write_csv(paths["evidence"], evidence_rows)
    write_csv(paths["efficiency"], efficiency_rows)
    write_jsonl(paths["per_sample"], all_per_sample_rows)
    write_json(
        paths["summary"],
        {
            "evaluated_methods": [row["method"] for row in main_rows],
            "skipped_methods": skipped_methods,
            "samples_per_method": {row["method"]: row["samples"] for row in main_rows},
            "outputs": {key: str(value) for key, value in paths.items()},
        },
    )

    print(json.dumps({"main_results": main_rows, "skipped_methods": skipped_methods}, ensure_ascii=False, indent=2))
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
