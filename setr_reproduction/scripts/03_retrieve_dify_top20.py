import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PID_PATTERN = re.compile(r"(?im)^\s*PID\s*:\s*(.+?)\s*$")
TITLE_PATTERN = re.compile(r"(?im)^\s*Title\s*:\s*(.+?)\s*$")
CONTENT_PATTERN = re.compile(r"(?ims)^\s*Content\s*:\s*(.*)$")


def load_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            values[key] = value
            os.environ.setdefault(key, value)
    return values


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def normalize_api_key(api_key: str) -> str:
    api_key = api_key.strip()
    if api_key.lower().startswith("bearer "):
        return api_key.split(None, 1)[1].strip()
    return api_key


def parse_bool(value: str, default: bool = False) -> bool:
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_json_env(name: str) -> Optional[Dict[str, Any]]:
    value = get_env(name)
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return parsed


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def existing_qids(path: Path) -> set:
    if not path.exists():
        return set()
    qids = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                qids.add(json.loads(line)["qid"])
            except Exception:
                continue
    return qids


def parse_chunk_content(content: str) -> Tuple[Optional[str], Optional[str], str]:
    pid_match = PID_PATTERN.search(content or "")
    title_match = TITLE_PATTERN.search(content or "")
    content_match = CONTENT_PATTERN.search(content or "")
    pid = pid_match.group(1).strip() if pid_match else None
    title = title_match.group(1).strip() if title_match else None
    text = content_match.group(1).strip() if content_match else (content or "").strip()
    return pid, title, text


def build_payload(question: str, top_k: int, score_threshold: float, score_threshold_enabled: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"query": question}
    retrieval_model = parse_json_env("DIFY_RETRIEVAL_MODEL_JSON")
    external_retrieval_model = parse_json_env("DIFY_EXTERNAL_RETRIEVAL_MODEL_JSON")

    if retrieval_model is not None:
        payload["retrieval_model"] = retrieval_model
    if external_retrieval_model is not None:
        payload["external_retrieval_model"] = external_retrieval_model
    elif retrieval_model is None:
        # Dify 官方检索文档示例使用 external_retrieval_model 承载 top_k/score_threshold。
        payload["external_retrieval_model"] = {
            "top_k": top_k,
            "score_threshold": score_threshold,
            "score_threshold_enabled": score_threshold_enabled,
        }
    return payload


def call_dify_retrieve(
    api_base_url: str,
    dataset_id: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}/datasets/{dataset_id}/retrieve"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {error_body}")
        except Exception as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(min(2 * attempt, 10))

    raise RuntimeError(f"Dify retrieve failed after {max_retries} attempts: {last_error}")


def normalize_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []
    for rank, record in enumerate(records, start=1):
        segment = record.get("segment") or {}
        document = segment.get("document") or {}
        raw_content = segment.get("content", "")
        pid, title, text = parse_chunk_content(raw_content)
        candidates.append(
            {
                "rank": rank,
                "pid": pid,
                "title": title,
                "text": text,
                "raw_content": raw_content,
                "score": record.get("score"),
                "segment_id": segment.get("id"),
                "document_id": segment.get("document_id"),
                "document_name": document.get("name"),
                "position": segment.get("position"),
                "tokens": segment.get("tokens"),
                "word_count": segment.get("word_count"),
                "parse_ok": bool(pid and title),
            }
        )
    return candidates


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


def summarize(candidate_path: Path, summary_json_path: Path, summary_csv_path: Path) -> Dict[str, Any]:
    rows = read_jsonl(candidate_path)
    total = len(rows)
    if total == 0:
        summary = {"samples": 0}
    else:
        recalls = [row["topk_metrics"]["evidence_recall"] for row in rows]
        precisions = [row["topk_metrics"]["evidence_precision"] for row in rows]
        all_hits = [row["topk_metrics"]["all_support_hit"] for row in rows]
        title_hits = [row["topk_metrics"]["gold_title_hit"] for row in rows]
        returned_counts = [row["actual_returned_count"] for row in rows]
        parse_fail_counts = [row["parse_failed_count"] for row in rows]
        latencies = [row["latency_seconds"] for row in rows if row.get("latency_seconds") is not None]
        summary = {
            "samples": total,
            "avg_returned_count": sum(returned_counts) / total,
            "less_than_top_k_count": sum(1 for row in rows if row["actual_returned_count"] < row["requested_top_k"]),
            "empty_retrieval_count": sum(1 for count in returned_counts if count == 0),
            "parse_failed_candidate_count": sum(parse_fail_counts),
            "parse_failed_query_count": sum(1 for count in parse_fail_counts if count > 0),
            "topk_evidence_recall": sum(recalls) / total,
            "topk_evidence_precision": sum(precisions) / total,
            "topk_gold_title_hit": sum(title_hits) / total,
            "topk_all_support_hit": sum(all_hits) / total,
            "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        }

    write_json(summary_json_path, summary)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k candidates for HotpotQA queries via Dify Knowledge API.")
    parser.add_argument("--env", default=".env", help="Path to .env file.")
    parser.add_argument("--queries", default="setr_reproduction/data/processed/eval/queries_500.jsonl")
    parser.add_argument("--output-dir", default="setr_reproduction/results/candidates")
    parser.add_argument("--limit", type=int, default=0, help="Limit queries for testing. 0 means all.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Print first payload and exit without calling Dify.")
    args = parser.parse_args()

    load_env(Path(args.env))
    api_base_url = get_env("DIFY_API_BASE_URL", "https://api.dify.ai/v1")
    api_key = normalize_api_key(get_env("DIFY_API_KEY"))
    dataset_id = get_env("DIFY_DATASET_ID")
    retrieval_label = get_env("DIFY_RETRIEVAL_LABEL", "hybrid")
    top_k = int(get_env("DIFY_TOP_K", "20"))
    score_threshold = float(get_env("DIFY_SCORE_THRESHOLD", "0") or 0)
    score_threshold_enabled = parse_bool(get_env("DIFY_SCORE_THRESHOLD_ENABLED", "false"))
    timeout = int(get_env("DIFY_REQUEST_TIMEOUT", "60"))
    max_retries = int(get_env("DIFY_MAX_RETRIES", "3"))
    sleep_seconds = float(get_env("DIFY_SLEEP_SECONDS", "0.2"))

    if not api_key or not dataset_id:
        raise SystemExit("请先在 .env 中填写 DIFY_API_KEY 和 DIFY_DATASET_ID。")

    queries = read_jsonl(Path(args.queries))
    if args.limit > 0:
        queries = queries[: args.limit]

    output_dir = Path(args.output_dir)
    candidate_path = output_dir / f"dify_{retrieval_label}_top{top_k}_500.jsonl"
    raw_path = output_dir / f"dify_{retrieval_label}_top{top_k}_500_raw.jsonl"
    summary_json_path = output_dir / f"dify_{retrieval_label}_top{top_k}_500_summary.json"
    summary_csv_path = output_dir / f"dify_{retrieval_label}_top{top_k}_500_summary.csv"

    if args.force:
        for path in [candidate_path, raw_path, summary_json_path, summary_csv_path]:
            if path.exists():
                path.unlink()

    done_qids = existing_qids(candidate_path)
    first_payload = build_payload(queries[0]["question"], top_k, score_threshold, score_threshold_enabled)
    if args.dry_run:
        print(json.dumps({"url": f"{api_base_url.rstrip('/')}/datasets/{dataset_id}/retrieve", "payload": first_payload}, ensure_ascii=False, indent=2))
        return

    start_time = time.time()
    processed = 0
    for index, query_row in enumerate(queries, start=1):
        qid = query_row["qid"]
        if qid in done_qids:
            continue
        question = query_row["question"]
        payload = build_payload(question, top_k, score_threshold, score_threshold_enabled)
        t0 = time.time()
        response = call_dify_retrieve(api_base_url, dataset_id, api_key, payload, timeout, max_retries)
        latency = time.time() - t0
        records = response.get("records") or []
        candidates = normalize_records(records)
        metrics = evidence_metrics(query_row.get("gold_titles", []), [item.get("title") for item in candidates])

        normalized_row = {
            "qid": qid,
            "sample_index": query_row.get("sample_index"),
            "question": question,
            "answer": query_row.get("answer"),
            "gold_titles": query_row.get("gold_titles", []),
            "retrieval_backend": "dify",
            "retrieval_label": retrieval_label,
            "requested_top_k": top_k,
            "actual_returned_count": len(candidates),
            "parse_failed_count": sum(1 for item in candidates if not item["parse_ok"]),
            "latency_seconds": latency,
            "topk_metrics": metrics,
            "candidates": candidates,
        }
        raw_row = {
            "qid": qid,
            "sample_index": query_row.get("sample_index"),
            "question": question,
            "payload": payload,
            "latency_seconds": latency,
            "response": response,
        }
        append_jsonl(candidate_path, normalized_row)
        append_jsonl(raw_path, raw_row)
        processed += 1

        if processed % 10 == 0 or processed == len(queries):
            elapsed = time.time() - start_time
            print(f"processed={processed} current_index={index}/{len(queries)} elapsed={elapsed:.1f}s")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    summary = summarize(candidate_path, summary_json_path, summary_csv_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(candidate_path)
    print(raw_path)
    print(summary_csv_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
