"""Generate qualitative case analysis for SETR reproduction step 13."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "tables"
SELECTION_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "selections"
CASE_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "cases"


METHODS = [
    "retrieval_top5",
    "bge_reranker_top5",
    "llm_listwise_top5",
    "setr_selection_only",
    "setr_cot",
    "setr_cot_iri",
]

METHOD_LABELS = {
    "retrieval_top5": "Retrieval Top-5",
    "bge_reranker_top5": "BGE-Reranker Top-5",
    "llm_listwise_top5": "LLM Listwise Top-5",
    "setr_selection_only": "SetR-Selection only",
    "setr_cot": "SetR-CoT",
    "setr_cot_iri": "SetR-CoT + IRI",
}

SELECTION_FILES = {
    "retrieval_top5": "retrieval_top5_hybrid_500.jsonl",
    "bge_reranker_top5": "bge_reranker_top5_hybrid_500.jsonl",
    "llm_listwise_top5": "llm_listwise_top5_hybrid_500.jsonl",
    "setr_selection_only": "setr_selection_only_hybrid_500.jsonl",
    "setr_cot": "setr_cot_hybrid_500.jsonl",
    "setr_cot_iri": "setr_cot_iri_hybrid_500.jsonl",
}

SETR_METHODS = {"setr_selection_only", "setr_cot", "setr_cot_iri"}
BASELINE_METHODS = {"retrieval_top5", "bge_reranker_top5", "llm_listwise_top5"}


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


def get_qid(row: Dict[str, Any]) -> str:
    qid = row.get("qid") or row.get("_id") or row.get("id")
    if qid is None:
        raise ValueError(f"Missing qid in row: {row}")
    return str(qid)


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def is_answer_correct(row: Dict[str, Any]) -> bool:
    return to_float(row.get("answer_em")) >= 1.0


def all_support_hit(row: Dict[str, Any]) -> bool:
    return to_float(row.get("all_support_hit")) >= 1.0


def index_eval_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    indexed: Dict[str, Dict[str, Dict[str, Any]]] = {method: {} for method in METHODS}
    for row in rows:
        method = row.get("method")
        if method not in indexed:
            continue
        qid = get_qid(row)
        indexed[method][qid] = row
    return indexed


def index_selection_rows() -> Dict[str, Dict[str, Dict[str, Any]]]:
    indexed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method, filename in SELECTION_FILES.items():
        path = SELECTION_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Selection file not found: {path}")
        indexed[method] = {get_qid(row): row for row in read_jsonl(path)}
    return indexed


def selected_titles(row: Dict[str, Any]) -> List[str]:
    return [str(item.get("title", "")) for item in row.get("selected_passages", []) if item.get("title")]


def get_metric_titles(selection_row: Dict[str, Any], key: str) -> List[str]:
    metrics = selection_row.get("selection_metrics") or {}
    return [str(item) for item in metrics.get(key, [])]


def get_top20_metrics(selection_row: Dict[str, Any]) -> Dict[str, Any]:
    return selection_row.get("top20_metrics") or {}


def build_method_snapshot(method: str, eval_row: Dict[str, Any], selection_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "generated_answer": eval_row.get("generated_answer"),
        "answer_em": eval_row.get("answer_em"),
        "answer_f1": eval_row.get("answer_f1"),
        "all_support_hit": eval_row.get("all_support_hit"),
        "evidence_recall": eval_row.get("evidence_recall"),
        "evidence_precision": eval_row.get("evidence_precision"),
        "selected_count": eval_row.get("selected_count"),
        "selected_titles": selected_titles(selection_row),
        "hit_gold_titles": get_metric_titles(selection_row, "hit_gold_titles"),
        "missing_gold_titles": get_metric_titles(selection_row, "missing_gold_titles"),
        "parse_success": selection_row.get("parse_success"),
        "fallback_used": selection_row.get("fallback_used", False),
    }


def build_case(
    case_id: str,
    category: str,
    qid: str,
    reason: str,
    eval_maps: Dict[str, Dict[str, Dict[str, Any]]],
    selection_maps: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    base_eval = eval_maps["retrieval_top5"][qid]
    base_selection = selection_maps["retrieval_top5"][qid]
    top20_metrics = get_top20_metrics(base_selection)
    methods = {
        method: build_method_snapshot(method, eval_maps[method][qid], selection_maps[method][qid])
        for method in METHODS
    }
    return {
        "case_id": case_id,
        "category": category,
        "qid": qid,
        "sample_index": base_eval.get("sample_index"),
        "question": base_eval.get("question"),
        "gold_answer": base_eval.get("gold_answer"),
        "gold_titles": base_selection.get("gold_titles", []),
        "top20_all_support_hit": top20_metrics.get("all_support_hit"),
        "top20_hit_gold_titles": top20_metrics.get("hit_gold_titles", []),
        "top20_missing_gold_titles": top20_metrics.get("missing_gold_titles", []),
        "reason": reason,
        "methods": methods,
    }


def qids(eval_maps: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    base = set(eval_maps[METHODS[0]])
    for method in METHODS[1:]:
        if set(eval_maps[method]) != base:
            raise ValueError(f"QID mismatch for {method}")
    return sorted(base, key=lambda qid: eval_maps["retrieval_top5"][qid].get("sample_index", 10**9))


def choose_first(candidates: List[str], used: set[str]) -> Optional[str]:
    for qid in candidates:
        if qid not in used:
            return qid
    return None


def select_cases(
    eval_maps: Dict[str, Dict[str, Dict[str, Any]]],
    selection_maps: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    used: set[str] = set()
    all_qids = qids(eval_maps)
    cases: List[Dict[str, Any]] = []

    # 1. SetR-CoT + IRI succeeds while top-5 style baselines fail.
    strict_iri_success = [
        qid
        for qid in all_qids
        if is_answer_correct(eval_maps["setr_cot_iri"][qid])
        and all_support_hit(eval_maps["setr_cot_iri"][qid])
        and not is_answer_correct(eval_maps["retrieval_top5"][qid])
        and not is_answer_correct(eval_maps["bge_reranker_top5"][qid])
        and not is_answer_correct(eval_maps["llm_listwise_top5"][qid])
    ]
    relaxed_iri_success = [
        qid
        for qid in all_qids
        if is_answer_correct(eval_maps["setr_cot_iri"][qid])
        and not is_answer_correct(eval_maps["retrieval_top5"][qid])
        and not is_answer_correct(eval_maps["llm_listwise_top5"][qid])
    ]
    selected = choose_first(strict_iri_success or relaxed_iri_success, used)
    if selected:
        used.add(selected)
        cases.append(
            build_case(
                "case_1_setr_iri_success_baseline_fail",
                "SetR-CoT + IRI 成功，Top-5 / reranker 失败",
                selected,
                "SetR-CoT + IRI 给出完全匹配答案，而至少主要 Top-5 baseline 未能给出 EM 正确答案；用于展示集合选择在部分样本上能压缩上下文并保留关键证据。",
                eval_maps,
                selection_maps,
            )
        )

    # 2. Reranker succeeds while SetR variants fail.
    reranker_success = [
        qid
        for qid in all_qids
        if (
            is_answer_correct(eval_maps["bge_reranker_top5"][qid])
            or is_answer_correct(eval_maps["llm_listwise_top5"][qid])
        )
        and not any(is_answer_correct(eval_maps[method][qid]) for method in SETR_METHODS)
    ]
    selected = choose_first(reranker_success, used)
    if selected:
        used.add(selected)
        cases.append(
            build_case(
                "case_2_reranker_success_setr_fail",
                "reranker 成功，SetR 失败",
                selected,
                "BGE 或 LLM Listwise Top-5 能答对，但三个 SetR 变体均未 EM 命中；用于说明过度压缩上下文或漏选证据会损伤答案生成。",
                eval_maps,
                selection_maps,
            )
        )

    # 3. All methods fail and the top-20 candidate pool misses full support.
    candidate_miss = [
        qid
        for qid in all_qids
        if not any(is_answer_correct(eval_maps[method][qid]) for method in METHODS)
        and to_float(get_top20_metrics(selection_maps["retrieval_top5"][qid]).get("all_support_hit")) < 1.0
    ]
    selected = choose_first(candidate_miss, used)
    if selected:
        used.add(selected)
        cases.append(
            build_case(
                "case_3_candidate_pool_missing_gold",
                "三者都失败，原因是 top-20 候选池没召回完整 gold evidence",
                selected,
                "所有方法均未 EM 命中，并且 Hybrid Top-20 本身没有覆盖全部 gold titles；用于说明 first-stage retrieval 上限会限制后续 rerank / SetR。",
                eval_maps,
                selection_maps,
            )
        )

    # 4. SetR answers correctly with fewer passages than Top-5.
    compact_setr_success = [
        qid
        for qid in all_qids
        if is_answer_correct(eval_maps["setr_selection_only"][qid])
        and to_float(eval_maps["setr_selection_only"][qid].get("selected_count")) < 5
        and all_support_hit(eval_maps["setr_selection_only"][qid])
    ]
    selected = choose_first(compact_setr_success, used)
    if selected:
        used.add(selected)
        cases.append(
            build_case(
                "case_4_setr_fewer_passages_correct",
                "SetR 选更少 passage 但答案正确",
                selected,
                "SetR-Selection only 用少于 5 个 passage 覆盖完整 gold evidence 并答对；用于展示 SetR 的上下文压缩能力。",
                eval_maps,
                selection_maps,
            )
        )

    # 5. Parse/fallback issue or redundant passage selection.
    parse_issue = [
        qid
        for qid in all_qids
        if any(selection_maps[method][qid].get("fallback_used") for method in SETR_METHODS)
        or any(selection_maps[method][qid].get("parse_success") is False for method in SETR_METHODS)
    ]
    redundant_iri = [
        qid
        for qid in all_qids
        if all_support_hit(eval_maps["setr_cot_iri"][qid])
        and to_float(eval_maps["setr_cot_iri"][qid].get("evidence_precision")) < 1.0
        and to_float(eval_maps["setr_cot_iri"][qid].get("selected_count")) > len(selection_maps["setr_cot_iri"][qid].get("gold_titles", []))
    ]
    selected = choose_first(parse_issue or redundant_iri, used)
    if selected:
        used.add(selected)
        reason = (
            "该样本触发 SetR 解析失败 / fallback，用于展示 prompt 输出格式问题。"
            if selected in parse_issue
            else "该样本虽覆盖 gold evidence，但额外选择了非 gold passage，用于展示 SetR 仍可能选择冗余上下文。"
        )
        cases.append(
            build_case(
                "case_5_parse_or_redundancy_issue",
                "Qwen3-14B 输出格式错误或选择冗余 passage",
                selected,
                reason,
                eval_maps,
                selection_maps,
            )
        )

    if len(cases) < 5:
        raise ValueError(f"Only selected {len(cases)} cases; expected 5.")
    return cases


def format_list(values: List[Any]) -> str:
    if not values:
        return "[]"
    return ", ".join(str(value) for value in values)


def method_table(case: Dict[str, Any]) -> str:
    lines = [
        "| Method | Generated answer | EM | F1 | All-support | Selected count | Selected titles | Missing gold titles |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for method in METHODS:
        row = case["methods"][method]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method_label"],
                    str(row.get("generated_answer", "")).replace("|", "\\|"),
                    str(row.get("answer_em")),
                    f"{to_float(row.get('answer_f1')):.4f}",
                    str(row.get("all_support_hit")),
                    str(row.get("selected_count")),
                    format_list(row.get("selected_titles", [])).replace("|", "\\|"),
                    format_list(row.get("missing_gold_titles", [])).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_case_markdown(cases: List[Dict[str, Any]]) -> str:
    lines = [
        "# 环节 13：案例分析",
        "",
        "本文件从 500 条 HotpotQA dev distractor 样本中选取 5 个典型案例，用于解释定量结果背后的原因。",
        "",
        "案例覆盖：SetR 成功、reranker 成功而 SetR 失败、候选池缺失、SetR 少选但答对、以及格式/冗余问题。",
        "",
    ]
    for index, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"## Case {index}: {case['category']}",
                "",
                f"- qid: `{case['qid']}`",
                f"- sample_index: {case['sample_index']}",
                f"- Question: {case['question']}",
                f"- Gold answer: {case['gold_answer']}",
                f"- Gold titles: {format_list(case['gold_titles'])}",
                f"- Top-20 all-support hit: {case['top20_all_support_hit']}",
                f"- Top-20 missing gold titles: {format_list(case['top20_missing_gold_titles'])}",
                "",
                method_table(case),
                "",
                "分析：",
                "",
                case["reason"],
                "",
            ]
        )
    lines.extend(
        [
            "## 总结",
            "",
            "这些案例说明：SetR 的主要优势是能够用更少 passage 保留回答所需信息；但当 first-stage retrieval 没有召回完整 gold evidence，或 prompt-level SetR 过度压缩上下文时，后续生成仍会失败。IRI 在部分样本上能帮助模型围绕信息需求选择证据，但在当前未微调设置下也可能引入冗余或漏选。"
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative case analysis.")
    parser.add_argument("--per-sample", default=str(TABLE_DIR / "per_sample_evaluation.jsonl"))
    parser.add_argument("--output-md", default=str(CASE_DIR / "case_analysis.md"))
    parser.add_argument("--output-jsonl", default=str(CASE_DIR / "case_analysis_cases.jsonl"))
    parser.add_argument("--summary-json", default=str(CASE_DIR / "case_analysis_summary.json"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def ensure_outputs(paths: List[Path], force: bool) -> None:
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
    per_sample_path = Path(args.per_sample)
    output_md = Path(args.output_md)
    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)

    if not per_sample_path.exists():
        raise FileNotFoundError(f"Per-sample evaluation file not found: {per_sample_path}")
    ensure_outputs([output_md, output_jsonl, summary_json], force=args.force)

    eval_maps = index_eval_rows(read_jsonl(per_sample_path))
    selection_maps = index_selection_rows()
    cases = select_cases(eval_maps, selection_maps)
    markdown = build_case_markdown(cases)
    summary = {
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "categories": [case["category"] for case in cases],
        "qids": [case["qid"] for case in cases],
        "outputs": {
            "case_analysis_md": str(output_md),
            "case_analysis_jsonl": str(output_jsonl),
            "summary_json": str(summary_json),
        },
    }

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    write_jsonl(output_jsonl, cases)
    write_json(summary_json, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_md)
    print(output_jsonl)
    print(summary_json)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
