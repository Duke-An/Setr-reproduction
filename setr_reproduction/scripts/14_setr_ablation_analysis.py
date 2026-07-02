"""Run SETR ablation analysis for reproduction step 11.

The analysis compares only the three SETR variants:

1. SetR-Selection only
2. SetR-CoT
3. SetR-CoT + IRI

It uses the unified step-10 evaluation outputs and the original selection
files to produce aggregate ablation tables, metric deltas, and per-question
case tags for later report writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTION_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "selections"
TABLE_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "tables"


SETR_METHODS = [
    "setr_selection_only",
    "setr_cot",
    "setr_cot_iri",
]

METHOD_LABELS = {
    "setr_selection_only": "SetR-Selection only",
    "setr_cot": "SetR-CoT",
    "setr_cot_iri": "SetR-CoT + IRI",
}

SELECTION_FILES = {
    "setr_selection_only": "setr_selection_only_hybrid_500.jsonl",
    "setr_cot": "setr_cot_hybrid_500.jsonl",
    "setr_cot_iri": "setr_cot_iri_hybrid_500.jsonl",
}

ABLATED_METRICS = [
    "answer_em",
    "answer_f1",
    "answer_contains",
    "evidence_recall",
    "evidence_precision",
    "all_support_hit",
    "avg_selected_passages",
    "avg_context_words",
    "selection_parse_failure_rate",
    "selection_fallback_rate",
    "avg_total_latency_seconds",
]


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


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_qid(row: Dict[str, Any]) -> str:
    qid = row.get("qid") or row.get("_id") or row.get("id")
    if qid is None:
        raise ValueError(f"Missing qid in row: {row}")
    return str(qid)


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def round4(value: float) -> float:
    return round(float(value), 4)


def selected_titles(row: Dict[str, Any]) -> List[str]:
    return [str(item.get("title", "")) for item in row.get("selected_passages", []) if item.get("title")]


def hit_titles(row: Dict[str, Any]) -> List[str]:
    metrics = row.get("selection_metrics") or {}
    return [str(item) for item in metrics.get("hit_gold_titles", [])]


def missing_titles(row: Dict[str, Any]) -> List[str]:
    metrics = row.get("selection_metrics") or {}
    return [str(item) for item in metrics.get("missing_gold_titles", [])]


def index_by_method_and_qid(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    indexed: Dict[str, Dict[str, Dict[str, Any]]] = {method: {} for method in SETR_METHODS}
    for row in rows:
        method = row.get("method")
        if method not in indexed:
            continue
        qid = get_qid(row)
        if qid in indexed[method]:
            raise ValueError(f"Duplicate qid in per-sample evaluation: method={method}, qid={qid}")
        indexed[method][qid] = row
    return indexed


def load_selection_maps() -> Dict[str, Dict[str, Dict[str, Any]]]:
    maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method, filename in SELECTION_FILES.items():
        path = SELECTION_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Selection file not found: {path}")
        method_map: Dict[str, Dict[str, Any]] = {}
        for row in read_jsonl(path):
            qid = get_qid(row)
            if qid in method_map:
                raise ValueError(f"Duplicate qid in selection file: method={method}, qid={qid}")
            method_map[qid] = row
        maps[method] = method_map
    return maps


def validate_same_qids(eval_maps: Dict[str, Dict[str, Dict[str, Any]]], expected_samples: int) -> List[str]:
    qid_sets = {method: set(eval_maps[method]) for method in SETR_METHODS}
    base_qids = qid_sets[SETR_METHODS[0]]
    for method in SETR_METHODS:
        if len(qid_sets[method]) != expected_samples:
            raise ValueError(f"{method} has {len(qid_sets[method])} samples, expected {expected_samples}")
        if qid_sets[method] != base_qids:
            missing = sorted(base_qids - qid_sets[method])[:5]
            extra = sorted(qid_sets[method] - base_qids)[:5]
            raise ValueError(f"QID mismatch for {method}: missing={missing}, extra={extra}")
    return sorted(base_qids)


def build_ablation_rows(main_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_method = {row["method"]: row for row in main_rows}
    missing = [method for method in SETR_METHODS if method not in by_method]
    if missing:
        raise ValueError(f"Missing SetR methods in main results: {missing}")

    rows: List[Dict[str, Any]] = []
    for index, method in enumerate(SETR_METHODS, start=1):
        source = by_method[method]
        row: Dict[str, Any] = {
            "order": index,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "samples": int(source["samples"]),
        }
        for metric in ABLATED_METRICS:
            row[metric] = round4(to_float(source[metric]))
        rows.append(row)
    return rows


def build_delta_rows(ablation_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method = {row["method"]: row for row in ablation_rows}
    comparisons = [
        ("cot_minus_selection_only", "SetR-CoT - SetR-Selection only", "setr_cot", "setr_selection_only"),
        ("iri_minus_cot", "SetR-CoT + IRI - SetR-CoT", "setr_cot_iri", "setr_cot"),
        ("iri_minus_selection_only", "SetR-CoT + IRI - SetR-Selection only", "setr_cot_iri", "setr_selection_only"),
    ]

    rows: List[Dict[str, Any]] = []
    for comparison_id, label, method_a, method_b in comparisons:
        row: Dict[str, Any] = {
            "comparison": comparison_id,
            "comparison_label": label,
        }
        for metric in ABLATED_METRICS:
            row[f"delta_{metric}"] = round4(to_float(by_method[method_a][metric]) - to_float(by_method[method_b][metric]))
        rows.append(row)
    return rows


def int_metric(row: Dict[str, Any], key: str) -> int:
    return int(round(to_float(row.get(key))))


def classify_case(rows_by_method: Dict[str, Dict[str, Any]]) -> List[str]:
    selection = rows_by_method["setr_selection_only"]
    cot = rows_by_method["setr_cot"]
    iri = rows_by_method["setr_cot_iri"]

    tags: List[str] = []

    if int_metric(selection, "all_support_hit") == 0 and int_metric(cot, "all_support_hit") == 1:
        tags.append("cot_helped_evidence")
    if int_metric(selection, "all_support_hit") == 1 and int_metric(cot, "all_support_hit") == 0:
        tags.append("cot_hurt_evidence")
    if int_metric(cot, "all_support_hit") == 0 and int_metric(iri, "all_support_hit") == 1:
        tags.append("iri_helped_evidence")
    if int_metric(cot, "all_support_hit") == 1 and int_metric(iri, "all_support_hit") == 0:
        tags.append("iri_hurt_evidence")

    if int_metric(selection, "answer_em") == 0 and int_metric(cot, "answer_em") == 1:
        tags.append("cot_helped_answer_em")
    if int_metric(selection, "answer_em") == 1 and int_metric(cot, "answer_em") == 0:
        tags.append("cot_hurt_answer_em")
    if int_metric(cot, "answer_em") == 0 and int_metric(iri, "answer_em") == 1:
        tags.append("iri_helped_answer_em")
    if int_metric(cot, "answer_em") == 1 and int_metric(iri, "answer_em") == 0:
        tags.append("iri_hurt_answer_em")

    if not tags:
        tags.append("same_or_minor_change")
    return tags


def build_case_rows(
    qids: List[str],
    eval_maps: Dict[str, Dict[str, Dict[str, Any]]],
    selection_maps: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    case_rows: List[Dict[str, Any]] = []
    tag_counter: Counter[str] = Counter()

    for qid in qids:
        rows_by_method = {method: eval_maps[method][qid] for method in SETR_METHODS}
        selections_by_method = {method: selection_maps[method][qid] for method in SETR_METHODS}
        tags = classify_case(rows_by_method)
        tag_counter.update(tags)

        base_eval = rows_by_method["setr_selection_only"]
        base_selection = selections_by_method["setr_selection_only"]

        row: Dict[str, Any] = {
            "qid": qid,
            "sample_index": base_eval.get("sample_index"),
            "question": base_eval.get("question"),
            "gold_answer": base_eval.get("gold_answer"),
            "gold_titles": base_selection.get("gold_titles", []),
            "case_tags": tags,
            "primary_case_tag": tags[0],
        }

        for method in SETR_METHODS:
            eval_row = rows_by_method[method]
            selection_row = selections_by_method[method]
            prefix = method.replace("setr_", "")
            row[f"{prefix}_generated_answer"] = eval_row.get("generated_answer")
            row[f"{prefix}_answer_em"] = eval_row.get("answer_em")
            row[f"{prefix}_answer_f1"] = eval_row.get("answer_f1")
            row[f"{prefix}_all_support_hit"] = eval_row.get("all_support_hit")
            row[f"{prefix}_evidence_recall"] = eval_row.get("evidence_recall")
            row[f"{prefix}_evidence_precision"] = eval_row.get("evidence_precision")
            row[f"{prefix}_selected_count"] = eval_row.get("selected_count")
            row[f"{prefix}_selected_titles"] = selected_titles(selection_row)
            row[f"{prefix}_hit_gold_titles"] = hit_titles(selection_row)
            row[f"{prefix}_missing_gold_titles"] = missing_titles(selection_row)

        case_rows.append(row)

    case_rows.sort(key=lambda item: (item.get("sample_index") is None, item.get("sample_index") or 0, item["qid"]))
    return case_rows, tag_counter


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_summary_markdown(
    ablation_rows: List[Dict[str, Any]],
    delta_rows: List[Dict[str, Any]],
    tag_counter: Counter[str],
    total_samples: int,
) -> str:
    by_method = {row["method"]: row for row in ablation_rows}
    cot_delta = next(row for row in delta_rows if row["comparison"] == "cot_minus_selection_only")
    iri_delta = next(row for row in delta_rows if row["comparison"] == "iri_minus_cot")

    summary_lines = [
        "# 环节 11：SetR 消融分析结果",
        "",
        "## 1. 总体结果",
        "",
        markdown_table(
            ablation_rows,
            [
                "method_label",
                "answer_em",
                "answer_f1",
                "evidence_recall",
                "evidence_precision",
                "all_support_hit",
                "avg_selected_passages",
                "avg_context_words",
                "avg_total_latency_seconds",
            ],
        ),
        "",
        "## 2. 增量变化",
        "",
        markdown_table(
            delta_rows,
            [
                "comparison_label",
                "delta_answer_em",
                "delta_answer_f1",
                "delta_evidence_recall",
                "delta_evidence_precision",
                "delta_all_support_hit",
                "delta_avg_selected_passages",
                "delta_avg_total_latency_seconds",
            ],
        ),
        "",
        "## 3. 样本级变化统计",
        "",
        "| case_tag | count | ratio |",
        "|---|---:|---:|",
    ]
    for tag, count in sorted(tag_counter.items()):
        summary_lines.append(f"| {tag} | {count} | {count / total_samples:.4f} |")

    summary_lines.extend(
        [
            "",
            "## 4. 阶段结论",
            "",
            (
                f"CoT 相比 Selection only 的 F1 变化为 {cot_delta['delta_answer_f1']}，"
                f"Evidence Recall 变化为 {cot_delta['delta_evidence_recall']}，"
                f"All-support Hit 变化为 {cot_delta['delta_all_support_hit']}。"
            ),
            "",
            (
                f"IRI 相比 CoT 的 F1 变化为 {iri_delta['delta_answer_f1']}，"
                f"Evidence Precision 变化为 {iri_delta['delta_evidence_precision']}，"
                f"All-support Hit 变化为 {iri_delta['delta_all_support_hit']}。"
            ),
            "",
            (
                "在当前 prompt-level 复现设置下，IRI 的主要影响是进一步减少平均选择 passage 数，"
                "并略微提高 evidence precision；但它降低了 evidence recall 和 all-support hit，"
                "说明显式信息需求识别在没有微调模型配合时可能使选择策略过于保守。"
            ),
            "",
            "三个 SetR 变体的最终答案质量差距很小，其中 SetR-Selection only 的 EM/F1 最高；因此报告中应把 IRI 写成“高精度但可能漏证据”的消融现象，而不是强行写成稳定正收益。",
            "",
            "## 5. 关键数值",
            "",
            f"- SetR-Selection only: EM={by_method['setr_selection_only']['answer_em']}, F1={by_method['setr_selection_only']['answer_f1']}, All-support Hit={by_method['setr_selection_only']['all_support_hit']}, Avg Selected={by_method['setr_selection_only']['avg_selected_passages']}",
            f"- SetR-CoT: EM={by_method['setr_cot']['answer_em']}, F1={by_method['setr_cot']['answer_f1']}, All-support Hit={by_method['setr_cot']['all_support_hit']}, Avg Selected={by_method['setr_cot']['avg_selected_passages']}",
            f"- SetR-CoT + IRI: EM={by_method['setr_cot_iri']['answer_em']}, F1={by_method['setr_cot_iri']['answer_f1']}, All-support Hit={by_method['setr_cot_iri']['all_support_hit']}, Avg Selected={by_method['setr_cot_iri']['avg_selected_passages']}",
            "",
        ]
    )
    return "\n".join(summary_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SETR ablation analysis.")
    parser.add_argument("--main-results", default=str(TABLE_DIR / "main_results.csv"))
    parser.add_argument("--per-sample", default=str(TABLE_DIR / "per_sample_evaluation.jsonl"))
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def ensure_outputs(force: bool) -> Dict[str, Path]:
    paths = {
        "ablation": TABLE_DIR / "setr_ablation.csv",
        "delta": TABLE_DIR / "setr_ablation_delta.csv",
        "cases": TABLE_DIR / "setr_ablation_cases.jsonl",
        "case_summary": TABLE_DIR / "setr_ablation_case_summary.json",
        "summary_md": TABLE_DIR / "setr_ablation_summary.md",
    }
    if not force:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --force to overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    return paths


def main() -> None:
    args = parse_args()
    main_results_path = Path(args.main_results)
    per_sample_path = Path(args.per_sample)
    if not main_results_path.exists():
        raise FileNotFoundError(f"Main results file not found: {main_results_path}")
    if not per_sample_path.exists():
        raise FileNotFoundError(f"Per-sample evaluation file not found: {per_sample_path}")

    paths = ensure_outputs(force=args.force)

    main_rows = read_csv(main_results_path)
    per_sample_rows = read_jsonl(per_sample_path)
    eval_maps = index_by_method_and_qid(per_sample_rows)
    qids = validate_same_qids(eval_maps, expected_samples=args.expected_samples)
    selection_maps = load_selection_maps()
    for method in SETR_METHODS:
        if set(selection_maps[method]) != set(qids):
            raise ValueError(f"Selection qids do not match evaluation qids for {method}")

    ablation_rows = build_ablation_rows(main_rows)
    delta_rows = build_delta_rows(ablation_rows)
    case_rows, tag_counter = build_case_rows(qids, eval_maps, selection_maps)
    case_summary = {
        "samples": len(qids),
        "methods": SETR_METHODS,
        "tag_counts": dict(sorted(tag_counter.items())),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    summary_md = build_summary_markdown(
        ablation_rows=ablation_rows,
        delta_rows=delta_rows,
        tag_counter=tag_counter,
        total_samples=len(qids),
    )

    write_csv(paths["ablation"], ablation_rows)
    write_csv(paths["delta"], delta_rows)
    write_jsonl(paths["cases"], case_rows)
    write_json(paths["case_summary"], case_summary)
    paths["summary_md"].write_text(summary_md, encoding="utf-8")

    print(json.dumps({"ablation": ablation_rows, "delta": delta_rows, "case_summary": case_summary}, ensure_ascii=False, indent=2))
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
