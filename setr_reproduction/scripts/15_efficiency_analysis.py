"""Run efficiency analysis for SETR reproduction step 12.

This script extends the unified step-10 efficiency table with context-token
estimates, reduction ratios, runtime estimates, and figures.

Note: the generation script records context words/chars, not tokenizer-specific
token counts. We therefore report `avg_context_tokens_est = avg_context_chars / 4`
as a transparent approximation for English text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "figures"


METHOD_ORDER = [
    "retrieval_top5",
    "bge_reranker_top5",
    "llm_listwise_top5",
    "setr_selection_only",
    "setr_cot",
    "setr_cot_iri",
]

SHORT_LABELS = {
    "retrieval_top5": "Retrieval",
    "bge_reranker_top5": "BGE",
    "llm_listwise_top5": "LLM Listwise",
    "setr_selection_only": "SetR Sel.",
    "setr_cot": "SetR CoT",
    "setr_cot_iri": "SetR IRI",
}

TOP5_METHODS = {"retrieval_top5", "bge_reranker_top5", "llm_listwise_top5"}
SETR_METHODS = {"setr_selection_only", "setr_cot", "setr_cot_iri"}


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


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def round4(value: float) -> float:
    return round(float(value), 4)


def ordered_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_method = {row["method"]: row for row in rows}
    missing = [method for method in METHOD_ORDER if method not in by_method]
    if missing:
        raise ValueError(f"Missing methods in main results: {missing}")
    return [by_method[method] for method in METHOD_ORDER]


def reduction_ratio(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return 1.0 - value / baseline


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator_x == 0 or denominator_y == 0:
        return 0.0
    return numerator / (denominator_x * denominator_y)


def build_efficiency_rows(main_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows = ordered_rows(main_rows)
    retrieval = next(row for row in rows if row["method"] == "retrieval_top5")
    retrieval_passages = to_float(retrieval["avg_selected_passages"])
    retrieval_tokens = to_float(retrieval["avg_context_chars"]) / 4.0
    retrieval_words = to_float(retrieval["avg_context_words"])
    retrieval_generation_latency = to_float(retrieval["avg_generation_latency_seconds"])

    enhanced_rows: List[Dict[str, Any]] = []
    for row in rows:
        method = row["method"]
        samples = int(to_float(row["samples"]))
        avg_selected = to_float(row["avg_selected_passages"])
        avg_words = to_float(row["avg_context_words"])
        avg_chars = to_float(row["avg_context_chars"])
        avg_tokens_est = avg_chars / 4.0
        avg_selection_latency = to_float(row["avg_selection_latency_seconds"])
        avg_generation_latency = to_float(row["avg_generation_latency_seconds"])
        avg_total_latency = to_float(row["avg_total_latency_seconds"])
        answer_f1 = to_float(row["answer_f1"])
        answer_em = to_float(row["answer_em"])

        enhanced_rows.append(
            {
                "method": method,
                "method_label": row["method_label"],
                "samples": samples,
                "answer_em": round4(answer_em),
                "answer_f1": round4(answer_f1),
                "evidence_recall": round4(to_float(row["evidence_recall"])),
                "evidence_precision": round4(to_float(row["evidence_precision"])),
                "all_support_hit": round4(to_float(row["all_support_hit"])),
                "avg_selected_passages": round4(avg_selected),
                "avg_context_words": round4(avg_words),
                "avg_context_chars": round4(avg_chars),
                "avg_context_tokens_est": round4(avg_tokens_est),
                "passage_reduction_vs_retrieval_top5": round4(reduction_ratio(avg_selected, retrieval_passages)),
                "word_reduction_vs_retrieval_top5": round4(reduction_ratio(avg_words, retrieval_words)),
                "token_reduction_vs_retrieval_top5": round4(reduction_ratio(avg_tokens_est, retrieval_tokens)),
                "avg_selection_latency_seconds": round4(avg_selection_latency),
                "avg_generation_latency_seconds": round4(avg_generation_latency),
                "avg_total_latency_seconds": round4(avg_total_latency),
                "estimated_total_runtime_seconds": round4(avg_total_latency * samples),
                "generation_latency_reduction_vs_retrieval_top5": round4(
                    reduction_ratio(avg_generation_latency, retrieval_generation_latency)
                ),
                "selection_parse_failure_rate": round4(to_float(row["selection_parse_failure_rate"])),
                "selection_fallback_rate": round4(to_float(row["selection_fallback_rate"])),
                "generation_parse_failure_rate": round4(to_float(row["generation_parse_failure_rate"])),
                "generation_failure_rate": round4(to_float(row["generation_failure_rate"])),
                "answer_f1_per_1k_tokens_est": round4(answer_f1 / avg_tokens_est * 1000 if avg_tokens_est else 0.0),
                "answer_em_per_1k_tokens_est": round4(answer_em / avg_tokens_est * 1000 if avg_tokens_est else 0.0),
            }
        )
    return enhanced_rows


def validate_efficiency_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    warnings: List[str] = []
    by_method = {row["method"]: row for row in rows}

    for method in TOP5_METHODS:
        avg_selected = to_float(by_method[method]["avg_selected_passages"])
        if abs(avg_selected - 5.0) > 1e-9:
            raise ValueError(f"Top-5 method should have avg_selected_passages=5: {method}={avg_selected}")

    for method in SETR_METHODS:
        avg_selected = to_float(by_method[method]["avg_selected_passages"])
        if avg_selected > 5.0 + 1e-9:
            raise ValueError(f"SetR method should have avg_selected_passages <= 5: {method}={avg_selected}")

    passages = [to_float(row["avg_selected_passages"]) for row in rows]
    tokens = [to_float(row["avg_context_tokens_est"]) for row in rows]
    correlation = pearson(passages, tokens)
    if correlation < 0.5:
        warnings.append(
            f"Passage/token correlation is lower than expected: {correlation:.4f}. "
            "Check context construction or token estimate."
        )

    return {
        "top5_avg_selected_passages_checked": True,
        "setr_avg_selected_passages_checked": True,
        "passage_token_pearson_correlation": round4(correlation),
        "warnings": warnings,
    }


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_summary_md(rows: List[Dict[str, Any]], validation: Dict[str, Any]) -> str:
    by_method = {row["method"]: row for row in rows}
    setr_selection = by_method["setr_selection_only"]
    setr_cot = by_method["setr_cot"]
    setr_iri = by_method["setr_cot_iri"]

    return "\n".join(
        [
            "# 环节 12：效率分析结果",
            "",
            "## 1. 效率主表",
            "",
            markdown_table(
                rows,
                [
                    "method_label",
                    "answer_f1",
                    "avg_selected_passages",
                    "avg_context_tokens_est",
                    "token_reduction_vs_retrieval_top5",
                    "avg_generation_latency_seconds",
                    "avg_total_latency_seconds",
                    "answer_f1_per_1k_tokens_est",
                ],
            ),
            "",
            "## 2. 验证结果",
            "",
            f"- Top-5 baseline 平均 passage 数校验：{validation['top5_avg_selected_passages_checked']}",
            f"- SetR 平均 passage 数校验：{validation['setr_avg_selected_passages_checked']}",
            f"- passage 数与估算 token 数 Pearson 相关系数：{validation['passage_token_pearson_correlation']}",
            f"- warnings：{validation['warnings']}",
            "",
            "## 3. 关键结论",
            "",
            (
                f"SetR-Selection only 平均选择 {setr_selection['avg_selected_passages']} 个 passage，"
                f"估算上下文 token 为 {setr_selection['avg_context_tokens_est']}，"
                f"相比 Retrieval Top-5 token 减少 {setr_selection['token_reduction_vs_retrieval_top5']}。"
            ),
            "",
            (
                f"SetR-CoT 平均选择 {setr_cot['avg_selected_passages']} 个 passage，"
                f"估算上下文 token 为 {setr_cot['avg_context_tokens_est']}，"
                f"相比 Retrieval Top-5 token 减少 {setr_cot['token_reduction_vs_retrieval_top5']}。"
            ),
            "",
            (
                f"SetR-CoT + IRI 平均选择 {setr_iri['avg_selected_passages']} 个 passage，"
                f"估算上下文 token 为 {setr_iri['avg_context_tokens_est']}，"
                f"相比 Retrieval Top-5 token 减少 {setr_iri['token_reduction_vs_retrieval_top5']}。"
            ),
            "",
            "SetR 系列显著减少输入上下文长度，且答案 F1 与 Top-5 baseline 相当或更高，说明集合选择在本实验中具备较好的上下文压缩效率。",
            "",
            "需要注意：BGE-Reranker 的 selection latency 明显高于其他方法；LLM/SetR 方法的 latency 来自远程 Qwen3-14B API，受服务端负载影响，报告中应把延迟结果作为实验环境下的观测值，而不是模型绝对性能结论。",
            "",
        ]
    )


def draw_bar_chart(rows: List[Dict[str, Any]], metric: str, ylabel: str, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [SHORT_LABELS[row["method"]] for row in rows]
    values = [to_float(row[metric]) for row in rows]
    colors = ["#7f8c8d", "#6c5ce7", "#0984e3", "#00b894", "#00cec9", "#fdcb6e"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_scatter(rows: List[Dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        ax.scatter(to_float(row["avg_context_tokens_est"]), to_float(row["answer_f1"]), s=70)
        ax.text(
            to_float(row["avg_context_tokens_est"]) + 4,
            to_float(row["answer_f1"]),
            SHORT_LABELS[row["method"]],
            fontsize=8,
        )
    ax.set_xlabel("Estimated context tokens")
    ax.set_ylabel("Answer F1")
    ax.set_title("Answer F1 vs. Context Tokens")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_figures(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    figure_paths = {
        "avg_passages": FIGURE_DIR / "avg_passages.png",
        "avg_tokens": FIGURE_DIR / "avg_tokens.png",
        "avg_total_latency": FIGURE_DIR / "avg_total_latency.png",
        "f1_vs_tokens": FIGURE_DIR / "f1_vs_tokens.png",
    }
    draw_bar_chart(
        rows,
        metric="avg_selected_passages",
        ylabel="Average selected passages",
        title="Average Selected Passages",
        output_path=figure_paths["avg_passages"],
    )
    draw_bar_chart(
        rows,
        metric="avg_context_tokens_est",
        ylabel="Estimated context tokens",
        title="Estimated Context Tokens",
        output_path=figure_paths["avg_tokens"],
    )
    draw_bar_chart(
        rows,
        metric="avg_total_latency_seconds",
        ylabel="Average total latency (s)",
        title="Average Total Latency",
        output_path=figure_paths["avg_total_latency"],
    )
    draw_scatter(rows, figure_paths["f1_vs_tokens"])
    return {key: str(value) for key, value in figure_paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run efficiency analysis for step 12.")
    parser.add_argument("--main-results", default=str(TABLE_DIR / "main_results.csv"))
    parser.add_argument("--output", default=str(TABLE_DIR / "efficiency_results.csv"))
    parser.add_argument("--summary-md", default=str(TABLE_DIR / "efficiency_analysis.md"))
    parser.add_argument("--summary-json", default=str(TABLE_DIR / "efficiency_analysis_summary.json"))
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    parser.add_argument("--force", action="store_true", help="Overwrite outputs.")
    return parser.parse_args()


def ensure_can_write(paths: List[Path], force: bool) -> None:
    if force:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Output files already exist. Use --force to overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def main() -> None:
    args = parse_args()
    main_results_path = Path(args.main_results)
    output_path = Path(args.output)
    summary_md_path = Path(args.summary_md)
    summary_json_path = Path(args.summary_json)

    if not main_results_path.exists():
        raise FileNotFoundError(f"Main results file not found: {main_results_path}")

    planned_outputs = [output_path, summary_md_path, summary_json_path]
    if not args.no_figures:
        planned_outputs.extend(
            [
                FIGURE_DIR / "avg_passages.png",
                FIGURE_DIR / "avg_tokens.png",
                FIGURE_DIR / "avg_total_latency.png",
                FIGURE_DIR / "f1_vs_tokens.png",
            ]
        )
    ensure_can_write(planned_outputs, force=args.force)

    rows = build_efficiency_rows(read_csv(main_results_path))
    validation = validate_efficiency_rows(rows)
    figures = {} if args.no_figures else draw_figures(rows)
    summary_md = build_summary_md(rows, validation)
    summary = {
        "samples_per_method": {row["method"]: row["samples"] for row in rows},
        "validation": validation,
        "figures": figures,
        "outputs": {
            "efficiency_results": str(output_path),
            "summary_md": str(summary_md_path),
            "summary_json": str(summary_json_path),
        },
    }

    write_csv(output_path, rows)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.write_text(summary_md, encoding="utf-8")
    write_json(summary_json_path, summary)

    print(json.dumps({"efficiency_results": rows, "validation": validation, "figures": figures}, ensure_ascii=False, indent=2))
    print(output_path)
    print(summary_md_path)
    print(summary_json_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
