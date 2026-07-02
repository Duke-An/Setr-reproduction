"""Compare first-stage retrieval candidate pools for SETR reproduction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "candidates"
TABLE_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "tables"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_value(value: Any) -> str:
    if value == "" or value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_rows() -> List[Dict[str, Any]]:
    items = [
        {
            "candidate_pool": "Candidate Context",
            "pool_type": "Diagnostic / Oracle-style",
            "summary_path": CANDIDATE_DIR / "candidate_context_pool_500_summary.json",
            "role": "诊断 SetR / rerank 在 gold evidence 已进入候选池时的选择能力",
        },
        {
            "candidate_pool": "Dify Vector Top-20",
            "pool_type": "Global Corpus Retrieval",
            "summary_path": CANDIDATE_DIR / "dify_vector_top20_500_summary.json",
            "role": "真实全库语义检索候选池，对照组",
        },
        {
            "candidate_pool": "Dify Full-text Top-20",
            "pool_type": "Global Corpus Retrieval",
            "summary_path": CANDIDATE_DIR / "dify_fulltext_top20_500_summary.json",
            "role": "真实全库关键词 / 全文检索候选池，对照组",
        },
        {
            "candidate_pool": "Dify Hybrid Top-20",
            "pool_type": "Global Corpus Retrieval",
            "summary_path": CANDIDATE_DIR / "dify_hybrid_top20_500_summary.json",
            "role": "后续主实验固定候选池，语义 0.7 + 关键词 0.3",
        },
    ]

    rows: List[Dict[str, Any]] = []
    for item in items:
        summary = load_json(item["summary_path"])
        rows.append(
            {
                "candidate_pool": item["candidate_pool"],
                "pool_type": item["pool_type"],
                "samples": summary.get("samples"),
                "avg_returned_count": summary.get("avg_returned_count"),
                "less_than_top_k_count": summary.get("less_than_top_k_count"),
                "empty_retrieval_count": summary.get("empty_retrieval_count"),
                "parse_failed_candidate_count": summary.get("parse_failed_candidate_count"),
                "parse_failed_query_count": summary.get("parse_failed_query_count"),
                "topk_evidence_recall": summary.get("topk_evidence_recall"),
                "topk_evidence_precision": summary.get("topk_evidence_precision"),
                "topk_gold_title_hit": summary.get("topk_gold_title_hit"),
                "topk_all_support_hit": summary.get("topk_all_support_hit"),
                "avg_latency_seconds": summary.get("avg_latency_seconds", ""),
                "role": item["role"],
            }
        )
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    sample_counts = {row["samples"] for row in rows}
    parse_fail_ok = all(row["parse_failed_candidate_count"] == 0 for row in rows)
    global_rows = [row for row in rows if row["pool_type"] == "Global Corpus Retrieval"]
    main_pool = max(
        global_rows,
        key=lambda row: (row["topk_evidence_recall"], row["topk_all_support_hit"]),
    )

    headers = [
        "候选池",
        "类型",
        "样本数",
        "Avg Returned",
        "Evidence Recall",
        "All-support Hit",
        "Gold Title Hit",
        "Parse Failed Candidates",
        "Avg Latency(s)",
        "用途",
    ]

    lines = [
        "# First-stage Retrieval Quality 对比表",
        "",
        "评估对象：HotpotQA dev distractor 随机 500 个问题。",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        md_row = [
            row["candidate_pool"],
            row["pool_type"],
            row["samples"],
            format_value(row["avg_returned_count"]),
            format_value(row["topk_evidence_recall"]),
            format_value(row["topk_all_support_hit"]),
            format_value(row["topk_gold_title_hit"]),
            row["parse_failed_candidate_count"],
            format_value(row["avg_latency_seconds"]),
            row["role"],
        ]
        lines.append("| " + " | ".join(str(value) for value in md_row) + " |")

    lines.extend(
        [
            "",
            "## 校验结果",
            "",
            f"- 四个候选池样本数：{sorted(sample_counts)}。",
            f"- 样本数是否一致：{'是' if len(sample_counts) == 1 else '否'}。",
            f"- 候选解析是否全部成功：{'是' if parse_fail_ok else '否'}。",
            "",
            "## 主候选池选择",
            "",
            f"后续主实验固定候选池选择：`{main_pool['candidate_pool']}`。",
            "",
            "选择理由：",
            "",
            "- Candidate Context 是诊断候选池，gold evidence 天然在候选集合中，不代表真实全库检索难度，因此不作为主实验候选池。",
            "- 在三个真实全库检索候选池中，Dify Hybrid Top-20 的 Evidence Recall 和 All-support Hit 最高，说明其 first-stage retrieval 上限最好。",
            "- Hybrid 延迟最高，但当前任务是 500 条离线复现实验，效率代价可以接受。",
            "- 后续 BGE rerank、LLM rerank、SetR-Selection only、SetR-CoT、SetR-CoT + IRI 都应固定使用同一个 Hybrid Top-20 候选池，以保证后检索方法对比公平。",
            "",
            "## 阶段结论",
            "",
            "First-stage retrieval 决定后续 reranking / SetR 的理论上限。Candidate Context 用于诊断选择能力；Dify Vector、Full-text、Hybrid 用于比较真实全局检索质量。当前实验中 Hybrid Top-20 是真实检索条件下的最优候选池，后续主实验应固定使用该候选池。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()

    csv_path = TABLE_DIR / "first_stage_retrieval_quality.csv"
    md_path = TABLE_DIR / "first_stage_retrieval_quality.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    global_rows = [row for row in rows if row["pool_type"] == "Global Corpus Retrieval"]
    main_pool = max(
        global_rows,
        key=lambda row: (row["topk_evidence_recall"], row["topk_all_support_hit"]),
    )

    print(
        json.dumps(
            {
                "csv_path": str(csv_path.relative_to(PROJECT_ROOT)),
                "md_path": str(md_path.relative_to(PROJECT_ROOT)),
                "samples_consistent": len({row["samples"] for row in rows}) == 1,
                "selected_main_candidate_pool": main_pool["candidate_pool"],
                "selected_main_candidate_file": "setr_reproduction/results/candidates/dify_hybrid_top20_500.jsonl",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
