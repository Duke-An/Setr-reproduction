"""Run Qwen3-14B listwise reranker baseline on Dify Hybrid Top-20 candidates.

This script implements SETR reproduction step 5:
LLM Listwise Reranker Top-5 baseline.

It calls an OpenAI-compatible chat completions API and supports an empty API key.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "setr_reproduction"
    / "results"
    / "candidates"
    / "dify_hybrid_top20_500.jsonl"
)
DEFAULT_PROMPT = PROJECT_ROOT / "setr_reproduction" / "prompts" / "llm_listwise_reranker.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "selections"
DEFAULT_RAW_OUTPUT_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "raw_outputs"


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


def candidate_text(candidate: Dict[str, Any]) -> str:
    title = candidate.get("title") or ""
    text = candidate.get("text") or ""
    if title and text:
        return f"Title: {title}\nContent: {text}"
    return text or title


def build_user_prompt(row: Dict[str, Any], max_passages: int) -> str:
    candidates = row.get("candidates", [])[:max_passages]
    parts = [
        f"Question:\n{row.get('question', '')}",
        "",
        "Candidate passages:",
    ]
    for index, candidate in enumerate(candidates, start=1):
        parts.append(f"[{index}]\n{candidate_text(candidate)}")
    parts.extend(
        [
            "",
            f"Return a JSON object whose ranking contains every integer from 1 to {len(candidates)} exactly once.",
            "Do not include any text outside the JSON object.",
        ]
    )
    return "\n\n".join(parts)


def call_chat_completion(
    api_base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> Tuple[str, Dict[str, Any], float]:
    url = f"{api_base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    last_error: Optional[Exception] = None
    start = time.time()
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content, data, time.time() - start
        except Exception as exc:  # noqa: BLE001 - keep retry behavior simple.
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Chat completion failed after {max_retries} attempts: {last_error}")


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

    matches = re.findall(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if not matches:
        return None
    for candidate in reversed(matches):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def normalize_ranking(values: List[Any], candidate_count: int) -> List[int]:
    ranking: List[int] = []
    seen = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= candidate_count and number not in seen:
            ranking.append(number)
            seen.add(number)
    for number in range(1, candidate_count + 1):
        if number not in seen:
            ranking.append(number)
    return ranking


def parse_ranking(text: str, candidate_count: int) -> Tuple[List[int], bool, str]:
    obj = extract_json_object(text)
    if obj and isinstance(obj.get("ranking"), list):
        ranking = normalize_ranking(obj["ranking"], candidate_count)
        return ranking, True, "json"

    # Fallback: parse all integers from the output.
    numbers = re.findall(r"\b\d+\b", text)
    if numbers:
        ranking = normalize_ranking(numbers, candidate_count)
        if ranking:
            return ranking, False, "regex"

    return list(range(1, candidate_count + 1)), False, "fallback_original_order"


def build_selected_passages(
    candidates: List[Dict[str, Any]],
    ranking: List[int],
    top_k: int,
    gold_titles: List[str],
) -> List[Dict[str, Any]]:
    gold_title_set = set(gold_titles)
    selected: List[Dict[str, Any]] = []
    for selection_rank, passage_number in enumerate(ranking[:top_k], start=1):
        candidate = candidates[passage_number - 1]
        title = candidate.get("title")
        selected.append(
            {
                "selection_rank": selection_rank,
                "llm_passage_number": passage_number,
                "original_rank": candidate.get("rank"),
                "pid": candidate.get("pid"),
                "title": title,
                "text": candidate.get("text"),
                "retrieval_score": candidate.get("score"),
                "parse_ok": candidate.get("parse_ok"),
                "is_gold_title": title in gold_title_set if title else False,
            }
        )
    return selected


def fallback_top5(row: Dict[str, Any], top_k: int) -> List[int]:
    candidate_count = len(row.get("candidates", []))
    return list(range(1, candidate_count + 1))[: max(candidate_count, top_k)]


def rerank_one_row(
    row: Dict[str, Any],
    api_base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_passages: int,
    top_k: int,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidates = row.get("candidates", [])[:max_passages]
    candidate_count = len(candidates)
    user_prompt = build_user_prompt(row, max_passages=max_passages)
    user_prompt = "/no_think\n\n" + user_prompt

    error_message = ""
    raw_content = ""
    raw_response: Dict[str, Any] = {}
    latency = 0.0
    parse_success = False
    parse_method = "not_called"

    try:
        raw_content, raw_response, latency = call_chat_completion(
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
        ranking, parse_success, parse_method = parse_ranking(raw_content, candidate_count)
    except Exception as exc:  # noqa: BLE001 - fallback is part of experiment logging.
        error_message = str(exc)
        ranking = fallback_top5(row, top_k=top_k)
        parse_success = False
        parse_method = "api_error_fallback_original_order"

    if len(ranking) < candidate_count:
        ranking = normalize_ranking(ranking, candidate_count)

    selected_passages = build_selected_passages(
        candidates=candidates,
        ranking=ranking,
        top_k=top_k,
        gold_titles=row.get("gold_titles", []),
    )
    metrics = evidence_metrics(row.get("gold_titles", []), [item.get("title") for item in selected_passages])

    selection_row = {
        "qid": row.get("qid"),
        "sample_index": row.get("sample_index"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "gold_titles": row.get("gold_titles", []),
        "method": "llm_listwise_top5",
        "candidate_pool": "dify_hybrid_top20_500",
        "model": model,
        "selected_k_requested": top_k,
        "candidate_count": candidate_count,
        "selected_count": len(selected_passages),
        "llm_ranking": ranking,
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
    selected_k = rows[0]["selected_k_requested"]
    parse_success_count = sum(1 for row in rows if row["parse_success"])
    fallback_count = sum(1 for row in rows if row["fallback_used"])
    parse_failed_selected_count = sum(
        1
        for row in rows
        for item in row["selected_passages"]
        if item.get("parse_ok") is False or not item.get("pid") or not item.get("title")
    )

    return {
        "samples": total,
        "method": "llm_listwise_top5",
        "candidate_pool": "dify_hybrid_top20_500",
        "model": model,
        "selected_k_requested": selected_k,
        "avg_selected_count": sum(selected_counts) / total,
        "min_selected_count": min(selected_counts),
        "max_selected_count": max(selected_counts),
        "not_equal_selected_k_count": sum(1 for count in selected_counts if count != selected_k),
        "empty_selection_count": sum(1 for count in selected_counts if count == 0),
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-14B listwise reranker baseline.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES), help="Input Dify Hybrid Top-20 JSONL.")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="System prompt file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Selection output directory.")
    parser.add_argument("--raw-output-dir", default=str(DEFAULT_RAW_OUTPUT_DIR), help="Raw LLM output directory.")
    parser.add_argument("--api-base-url", default=os.getenv("LLM_API_BASE_URL", "http://YOUR_QWEN_API_HOST:PORT/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""), help="Empty means no Authorization header.")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "Qwen3-14B"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("LLM_MAX_TOKENS", "512")))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("LLM_TIMEOUT", "120")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("LLM_MAX_RETRIES", "2")))
    parser.add_argument("--max-passages", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Only process first N samples. 0 means all.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidates)
    prompt_path = Path(args.prompt)
    output_dir = Path(args.output_dir)
    raw_output_dir = Path(args.raw_output_dir)

    output_path = output_dir / "llm_listwise_top5_hybrid_500.jsonl"
    summary_json_path = output_dir / "llm_listwise_top5_hybrid_500_summary.json"
    summary_csv_path = output_dir / "llm_listwise_top5_hybrid_500_summary.csv"
    raw_output_path = raw_output_dir / "llm_listwise_outputs_hybrid_500.jsonl"

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
    candidate_rows = read_jsonl(candidate_path)
    if args.limit > 0:
        candidate_rows = candidate_rows[: args.limit]

    selection_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    start = time.time()

    for index, row in enumerate(candidate_rows, start=1):
        selection_row, raw_row = rerank_one_row(
            row=row,
            api_base_url=args.api_base_url,
            api_key=args.api_key.strip(),
            model=args.model,
            system_prompt=system_prompt,
            max_passages=args.max_passages,
            top_k=args.top_k,
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

    write_jsonl(output_path, selection_rows)
    write_jsonl(raw_output_path, raw_rows)
    write_json(summary_json_path, summary)
    write_summary_csv(summary_csv_path, summary)

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

