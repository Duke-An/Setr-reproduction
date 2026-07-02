"""Run Qwen3-14B SetR-CoT + IRI on Dify Hybrid Top-20 candidates.

This script implements SETR reproduction step 8:
prompt-level set selection with explicit Chain-of-Thought and
Information Requirement Identification (IRI).

It reuses common IO, API, and metric helpers from 10_run_setr_cot.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "setr_reproduction"
    / "results"
    / "candidates"
    / "dify_hybrid_top20_500.jsonl"
)
DEFAULT_PROMPT = PROJECT_ROOT / "setr_reproduction" / "prompts" / "setr_cot_iri.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "selections"
DEFAULT_RAW_OUTPUT_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "raw_outputs"


def load_common_module() -> Any:
    common_path = SCRIPT_DIR / "10_run_setr_cot.py"
    spec = importlib.util.spec_from_file_location("setr_cot_common", common_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load common module from {common_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load_common_module()


def build_user_prompt(row: Dict[str, Any], max_passages: int, max_selected: int) -> str:
    candidates = row.get("candidates", [])[:max_passages]
    parts = [
        f"Question:\n{row.get('question', '')}",
        "",
        "Candidate passages:",
    ]
    for index, candidate in enumerate(candidates, start=1):
        parts.append(f"[{index}]\n{COMMON.candidate_text(candidate)}")
    parts.extend(
        [
            "",
            "First identify the information requirements needed to answer the question.",
            "Then map each requirement to helpful passage numbers.",
            f"Finally return selected passages, at most {max_selected} integers.",
            f"Only use passage numbers from 1 to {len(candidates)}.",
            "Keep all fields concise. Do not include any text outside the JSON object.",
        ]
    )
    return "\n\n".join(parts)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Robustly find balanced JSON object candidates, because requirement_passages
    # may itself contain nested braces.
    candidates: List[str] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(cleaned):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(cleaned[start : index + 1])
                start = None

    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def normalize_selection(values: List[Any], candidate_count: int, max_selected: int) -> List[int]:
    selected: List[int] = []
    seen = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= candidate_count and number not in seen:
            selected.append(number)
            seen.add(number)
        if len(selected) >= max_selected:
            break
    return selected


def normalize_requirements(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_requirement_passages(value: Any, candidate_count: int, max_selected: int) -> Dict[str, List[int]]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, List[int]] = {}
    for key, passages in value.items():
        if isinstance(passages, list):
            normalized[str(key)] = normalize_selection(passages, candidate_count, max_selected)
        else:
            normalized[str(key)] = normalize_selection([passages], candidate_count, max_selected)
    return normalized


def parse_selection(
    text: str,
    candidate_count: int,
    max_selected: int,
) -> Tuple[List[int], List[str], Dict[str, List[int]], List[str], bool, str]:
    obj = extract_json_object(text)
    requirements: List[str] = []
    requirement_passages: Dict[str, List[int]] = {}
    reasoning_steps: List[str] = []

    if obj:
        requirements = normalize_requirements(obj.get("information_requirements"))
        requirement_passages = normalize_requirement_passages(
            obj.get("requirement_passages"),
            candidate_count=candidate_count,
            max_selected=max_selected,
        )
        reasoning_steps = normalize_requirements(obj.get("reasoning_steps"))

        if isinstance(obj.get("selected"), list):
            selected = normalize_selection(obj["selected"], candidate_count, max_selected)
            if selected:
                return selected, requirements, requirement_passages, reasoning_steps, True, "json"

        # If selected is absent but requirement_passages is usable, derive a compact
        # selection from the mapped passages.
        derived: List[int] = []
        for numbers in requirement_passages.values():
            for number in numbers:
                if number not in derived:
                    derived.append(number)
                if len(derived) >= max_selected:
                    break
            if len(derived) >= max_selected:
                break
        if derived:
            return derived, requirements, requirement_passages, reasoning_steps, False, "derived_from_requirement_passages"

    numbers = re.findall(r"\b\d+\b", text)
    if numbers:
        selected = normalize_selection(numbers, candidate_count, max_selected)
        if selected:
            return selected, requirements, requirement_passages, reasoning_steps, False, "regex"

    fallback = list(range(1, min(candidate_count, max_selected) + 1))
    return fallback, requirements, requirement_passages, reasoning_steps, False, "fallback_original_topk"


def select_one_row(
    row: Dict[str, Any],
    api_base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_passages: int,
    max_selected: int,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidates = row.get("candidates", [])[:max_passages]
    candidate_count = len(candidates)
    user_prompt = "/no_think\n\n" + build_user_prompt(
        row,
        max_passages=max_passages,
        max_selected=max_selected,
    )

    error_message = ""
    raw_content = ""
    raw_response: Dict[str, Any] = {}
    latency = 0.0
    parse_success = False
    parse_method = "not_called"
    information_requirements: List[str] = []
    requirement_passages: Dict[str, List[int]] = {}
    reasoning_steps: List[str] = []

    try:
        raw_content, raw_response, latency = COMMON.call_chat_completion(
            api_base_url=api_base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        (
            selected_numbers,
            information_requirements,
            requirement_passages,
            reasoning_steps,
            parse_success,
            parse_method,
        ) = parse_selection(
            raw_content,
            candidate_count=candidate_count,
            max_selected=max_selected,
        )
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        selected_numbers = list(range(1, min(candidate_count, max_selected) + 1))
        parse_success = False
        parse_method = "api_error_fallback_original_topk"

    selected_passages = COMMON.build_selected_passages(
        candidates=candidates,
        selected_numbers=selected_numbers,
        gold_titles=row.get("gold_titles", []),
    )
    metrics = COMMON.evidence_metrics(row.get("gold_titles", []), [item.get("title") for item in selected_passages])

    selection_row = {
        "qid": row.get("qid"),
        "sample_index": row.get("sample_index"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "gold_titles": row.get("gold_titles", []),
        "method": "setr_cot_iri",
        "candidate_pool": "dify_hybrid_top20_500",
        "model": model,
        "max_selected_requested": max_selected,
        "candidate_count": candidate_count,
        "selected_count": len(selected_passages),
        "llm_selected": selected_numbers,
        "information_requirements": information_requirements,
        "information_requirement_count": len(information_requirements),
        "requirement_passages": requirement_passages,
        "mapped_requirement_count": sum(1 for passages in requirement_passages.values() if passages),
        "reasoning_steps": reasoning_steps,
        "reasoning_step_count": len(reasoning_steps),
        "reasoning_char_count": len(json.dumps(reasoning_steps, ensure_ascii=False)),
        "parse_success": parse_success,
        "parse_method": parse_method,
        "fallback_used": not parse_success,
        "selection_latency_seconds": latency,
        "selection_metrics": metrics,
        "top20_metrics": row.get("topk_metrics"),
        "selected_passages": selected_passages,
    }

    raw_row = {
        "qid": row.get("qid"),
        "sample_index": row.get("sample_index"),
        "model": model,
        "parse_success": parse_success,
        "parse_method": parse_method,
        "fallback_used": not parse_success,
        "error_message": error_message,
        "latency_seconds": latency,
        "prompt": user_prompt,
        "raw_content": raw_content,
        "raw_response": raw_response,
    }
    return selection_row, raw_row


def summarize(rows: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"samples": 0}

    selected_counts = [row["selected_count"] for row in rows]
    recalls = [row["selection_metrics"]["evidence_recall"] for row in rows]
    precisions = [row["selection_metrics"]["evidence_precision"] for row in rows]
    title_hits = [row["selection_metrics"]["gold_title_hit"] for row in rows]
    all_hits = [row["selection_metrics"]["all_support_hit"] for row in rows]
    latencies = [row["selection_latency_seconds"] for row in rows]
    reasoning_chars = [row["reasoning_char_count"] for row in rows]
    reasoning_steps = [row["reasoning_step_count"] for row in rows]
    requirement_counts = [row["information_requirement_count"] for row in rows]
    mapped_requirement_counts = [row["mapped_requirement_count"] for row in rows]
    parse_success_count = sum(1 for row in rows if row["parse_success"])
    fallback_count = sum(1 for row in rows if row["fallback_used"])
    count_distribution = dict(sorted(Counter(selected_counts).items()))
    parse_failed_selected_count = sum(
        1
        for row in rows
        for item in row["selected_passages"]
        if item.get("parse_ok") is False or not item.get("pid") or not item.get("title")
    )

    return {
        "samples": total,
        "method": "setr_cot_iri",
        "candidate_pool": "dify_hybrid_top20_500",
        "model": model,
        "max_selected_requested": rows[0]["max_selected_requested"],
        "avg_selected_count": sum(selected_counts) / total,
        "min_selected_count": min(selected_counts),
        "max_selected_count": max(selected_counts),
        "selected_count_distribution": json.dumps(count_distribution, ensure_ascii=False),
        "empty_selection_count": sum(1 for count in selected_counts if count == 0),
        "over_max_selected_count": sum(1 for count in selected_counts if count > rows[0]["max_selected_requested"]),
        "parse_success_count": parse_success_count,
        "parse_success_rate": parse_success_count / total,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / total,
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
        "avg_information_requirement_count": sum(requirement_counts) / total,
        "avg_mapped_requirement_count": sum(mapped_requirement_counts) / total,
        "avg_reasoning_step_count": sum(reasoning_steps) / total,
        "avg_reasoning_char_count": sum(reasoning_chars) / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-14B SetR-CoT + IRI.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES), help="Input Dify Hybrid Top-20 JSONL.")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="System prompt file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Selection output directory.")
    parser.add_argument("--raw-output-dir", default=str(DEFAULT_RAW_OUTPUT_DIR), help="Raw LLM output directory.")
    parser.add_argument("--api-base-url", default=os.getenv("LLM_API_BASE_URL", "http://YOUR_QWEN_API_HOST:PORT/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""), help="Empty means no Authorization header.")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "Qwen3-14B"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("LLM_MAX_TOKENS", "1024")))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("LLM_TIMEOUT", "120")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("LLM_MAX_RETRIES", "2")))
    parser.add_argument("--max-passages", type=int, default=20)
    parser.add_argument("--max-selected", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Only process first N samples. 0 means all.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidates)
    prompt_path = Path(args.prompt)
    output_dir = Path(args.output_dir)
    raw_output_dir = Path(args.raw_output_dir)

    output_path = output_dir / "setr_cot_iri_hybrid_500.jsonl"
    summary_json_path = output_dir / "setr_cot_iri_hybrid_500_summary.json"
    summary_csv_path = output_dir / "setr_cot_iri_hybrid_500_summary.csv"
    raw_output_path = raw_output_dir / "setr_cot_iri_outputs_hybrid_500.jsonl"

    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    if not args.force:
        existing = [
            path
            for path in [output_path, summary_json_path, summary_csv_path, raw_output_path]
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --force to overwrite: "
                + ", ".join(str(path) for path in existing)
            )

    system_prompt = prompt_path.read_text(encoding="utf-8")
    candidate_rows = COMMON.read_jsonl(candidate_path)
    if args.limit > 0:
        candidate_rows = candidate_rows[: args.limit]

    selection_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    start = time.time()

    for index, row in enumerate(candidate_rows, start=1):
        selection_row, raw_row = select_one_row(
            row=row,
            api_base_url=args.api_base_url,
            api_key=args.api_key.strip(),
            model=args.model,
            system_prompt=system_prompt,
            max_passages=args.max_passages,
            max_selected=args.max_selected,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        selection_rows.append(selection_row)
        raw_rows.append(raw_row)
        if index % 10 == 0 or index == len(candidate_rows):
            elapsed = time.time() - start
            print(f"processed={index}/{len(candidate_rows)} elapsed={elapsed:.1f}s", flush=True)

    summary = summarize(selection_rows, model=args.model)

    COMMON.write_jsonl(output_path, selection_rows)
    COMMON.write_jsonl(raw_output_path, raw_rows)
    COMMON.write_json(summary_json_path, summary)
    COMMON.write_summary_csv(summary_csv_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_path)
    print(raw_output_path)
    print(summary_json_path)
    print(summary_csv_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

