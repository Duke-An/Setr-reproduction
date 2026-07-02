"""Generate answers from selected passages for SETR reproduction step 9.

This script uses a single Qwen3-14B generator for all methods so that
end-to-end QA differences mainly reflect evidence selection quality.
It calls an OpenAI-compatible /chat/completions endpoint and supports no API key.
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
SELECTION_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "selections"
GENERATION_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "generations"
RAW_OUTPUT_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "raw_outputs"
TABLE_DIR = PROJECT_ROOT / "setr_reproduction" / "results" / "tables"
DEFAULT_PROMPT = PROJECT_ROOT / "setr_reproduction" / "prompts" / "answer_generation.txt"


METHOD_FILES = {
    "retrieval_top5": "retrieval_top5_hybrid_500.jsonl",
    "bge_reranker_top5": "bge_reranker_top5_hybrid_500.jsonl",
    "llm_listwise_top5": "llm_listwise_top5_hybrid_500.jsonl",
    "setr_selection_only": "setr_selection_only_hybrid_500.jsonl",
    "setr_cot": "setr_cot_hybrid_500.jsonl",
    "setr_cot_iri": "setr_cot_iri_hybrid_500.jsonl",
}


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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def context_word_count(passages: List[Dict[str, Any]]) -> int:
    return sum(len((item.get("text") or "").split()) for item in passages)


def context_char_count(passages: List[Dict[str, Any]]) -> int:
    return sum(len(item.get("text") or "") for item in passages)


def build_context(passages: List[Dict[str, Any]], max_context_chars: int) -> Tuple[str, bool]:
    parts: List[str] = []
    truncated = False
    used_chars = 0

    for index, passage in enumerate(passages, start=1):
        title = normalize_space(passage.get("title") or "")
        text = normalize_space(passage.get("text") or "")
        block = f"[{index}] Title: {title}\nContent: {text}"
        remaining = max_context_chars - used_chars
        if remaining <= 0:
            truncated = True
            break
        if len(block) > remaining:
            block = block[:remaining]
            truncated = True
        parts.append(block)
        used_chars += len(block)

    return "\n\n".join(parts), truncated


def build_user_prompt(row: Dict[str, Any], max_context_chars: int) -> Tuple[str, bool, int, int]:
    passages = row.get("selected_passages", [])
    context, truncated = build_context(passages, max_context_chars=max_context_chars)
    user_prompt = "\n\n".join(
        [
            "/no_think",
            f"Question:\n{row.get('question', '')}",
            "Passages:",
            context if context else "(No passages selected.)",
            "Return JSON only.",
        ]
    )
    return user_prompt, truncated, context_word_count(passages), context_char_count(passages)


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
        except Exception as exc:  # noqa: BLE001
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
    for candidate in reversed(matches):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def parse_answer(raw_content: str) -> Tuple[str, bool, str]:
    obj = extract_json_object(raw_content)
    if obj and "answer" in obj:
        answer = normalize_space(str(obj["answer"]))
        if answer:
            return answer, True, "json"

    fallback_answer = normalize_space(raw_content)
    if fallback_answer:
        # Remove common wrappers if the model ignored JSON.
        fallback_answer = re.sub(r"^answer\s*:\s*", "", fallback_answer, flags=re.IGNORECASE)
        return fallback_answer, False, "raw_text"

    return "", False, "empty"


def generate_one(
    row: Dict[str, Any],
    method: str,
    api_base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    max_context_chars: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    user_prompt, context_truncated, word_count, char_count = build_user_prompt(
        row,
        max_context_chars=max_context_chars,
    )

    raw_content = ""
    raw_response: Dict[str, Any] = {}
    latency = 0.0
    error_message = ""
    parse_success = False
    parse_method = "not_called"
    generated_answer = ""

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
        generated_answer, parse_success, parse_method = parse_answer(raw_content)
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        generated_answer = ""
        parse_success = False
        parse_method = "api_error"

    generation_row = {
        "qid": row.get("qid"),
        "sample_index": row.get("sample_index"),
        "method": method,
        "model": model,
        "question": row.get("question"),
        "gold_answer": row.get("answer"),
        "generated_answer": generated_answer,
        "selected_count": row.get("selected_count"),
        "selected_titles": [item.get("title") for item in row.get("selected_passages", [])],
        "context_word_count": word_count,
        "context_char_count": char_count,
        "context_truncated": context_truncated,
        "generation_latency_seconds": latency,
        "parse_success": parse_success,
        "parse_method": parse_method,
        "generation_failed": bool(error_message) or not bool(generated_answer),
        "error_message": error_message,
    }

    raw_row = {
        "qid": row.get("qid"),
        "sample_index": row.get("sample_index"),
        "method": method,
        "model": model,
        "parse_success": parse_success,
        "parse_method": parse_method,
        "generation_failed": generation_row["generation_failed"],
        "error_message": error_message,
        "latency_seconds": latency,
        "prompt": user_prompt,
        "raw_content": raw_content,
        "raw_response": raw_response,
    }
    return generation_row, raw_row


def summarize_generation(rows: List[Dict[str, Any]], method: str, model: str) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"method": method, "samples": 0}

    latencies = [row["generation_latency_seconds"] for row in rows]
    selected_counts = [row["selected_count"] for row in rows if row.get("selected_count") is not None]
    word_counts = [row["context_word_count"] for row in rows]
    char_counts = [row["context_char_count"] for row in rows]
    parse_success_count = sum(1 for row in rows if row["parse_success"])
    failure_count = sum(1 for row in rows if row["generation_failed"])

    return {
        "method": method,
        "samples": total,
        "model": model,
        "parse_success_count": parse_success_count,
        "parse_success_rate": parse_success_count / total,
        "generation_failure_count": failure_count,
        "generation_failure_rate": failure_count / total,
        "avg_selected_count": sum(selected_counts) / len(selected_counts) if selected_counts else 0,
        "avg_context_word_count": sum(word_counts) / total,
        "avg_context_char_count": sum(char_counts) / total,
        "context_truncated_count": sum(1 for row in rows if row["context_truncated"]),
        "avg_generation_latency_seconds": sum(latencies) / total,
    }


def parse_methods(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(METHOD_FILES.keys())
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHOD_FILES]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {list(METHOD_FILES)}")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate answers for selected passages.")
    parser.add_argument("--methods", default="all", help="Comma-separated methods or 'all'.")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="Answer generation system prompt.")
    parser.add_argument("--api-base-url", default=os.getenv("LLM_API_BASE_URL", "http://YOUR_QWEN_API_HOST:PORT/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""), help="Empty means no Authorization header.")
    parser.add_argument("--model", default=os.getenv("GENERATOR_MODEL", os.getenv("LLM_MODEL", "Qwen3-14B")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("GENERATOR_TEMPERATURE", "0")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("GENERATOR_MAX_TOKENS", "128")))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("GENERATOR_TIMEOUT", "120")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("GENERATOR_MAX_RETRIES", "2")))
    parser.add_argument("--max-context-chars", type=int, default=int(os.getenv("GENERATOR_MAX_CONTEXT_CHARS", "24000")))
    parser.add_argument("--limit", type=int, default=0, help="Only process first N samples per method. 0 means all.")
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow non-500 selection files in full run.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generation outputs.")
    return parser.parse_args()


def validate_selection_rows(method: str, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if args.limit == 0 and not args.allow_incomplete and len(rows) != args.expected_samples:
        raise ValueError(
            f"Selection file for method={method} has {len(rows)} rows, "
            f"expected {args.expected_samples}. Use --allow-incomplete only for debugging."
        )


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    system_prompt = prompt_path.read_text(encoding="utf-8")

    summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        selection_path = SELECTION_DIR / METHOD_FILES[method]
        if not selection_path.exists():
            raise FileNotFoundError(f"Selection file not found for {method}: {selection_path}")

        selection_rows = read_jsonl(selection_path)
        validate_selection_rows(method, selection_rows, args)
        if args.limit > 0:
            selection_rows = selection_rows[: args.limit]

        generation_path = GENERATION_DIR / f"{method}_hybrid_500_answers.jsonl"
        raw_path = RAW_OUTPUT_DIR / f"{method}_answer_generation_outputs_hybrid_500.jsonl"
        summary_path = GENERATION_DIR / f"{method}_hybrid_500_generation_summary.json"
        summary_csv_path = GENERATION_DIR / f"{method}_hybrid_500_generation_summary.csv"

        if not args.force:
            existing = [path for path in [generation_path, raw_path, summary_path, summary_csv_path] if path.exists()]
            if existing:
                raise FileExistsError(
                    "Output files already exist. Use --force to overwrite: "
                    + ", ".join(str(path) for path in existing)
                )

        generation_rows: List[Dict[str, Any]] = []
        raw_rows: List[Dict[str, Any]] = []
        start = time.time()
        for index, row in enumerate(selection_rows, start=1):
            generation_row, raw_row = generate_one(
                row=row,
                method=method,
                api_base_url=args.api_base_url,
                api_key=args.api_key.strip(),
                model=args.model,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_context_chars=args.max_context_chars,
            )
            generation_rows.append(generation_row)
            raw_rows.append(raw_row)
            if index % 10 == 0 or index == len(selection_rows):
                elapsed = time.time() - start
                print(f"method={method} processed={index}/{len(selection_rows)} elapsed={elapsed:.1f}s", flush=True)

        summary = summarize_generation(generation_rows, method=method, model=args.model)
        summary_rows.append(summary)

        write_jsonl(generation_path, generation_rows)
        write_jsonl(raw_path, raw_rows)
        write_json(summary_path, summary)
        write_csv(summary_csv_path, [summary])

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(generation_path)
        print(raw_path)

    combined_summary_path = TABLE_DIR / "generation_summary.csv"
    write_csv(combined_summary_path, summary_rows)
    print(combined_summary_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

