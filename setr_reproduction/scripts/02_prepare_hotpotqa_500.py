import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def unique_in_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_gold_titles(item: Dict[str, Any]) -> List[str]:
    return unique_in_order(normalize_space(fact[0]) for fact in item.get("supporting_facts", []) if fact)


def build_candidate_contexts(item: Dict[str, Any], gold_titles: List[str]) -> List[Dict[str, Any]]:
    qid = item["_id"]
    gold_title_set = set(gold_titles)
    candidates = []
    for index, context in enumerate(item.get("context", [])):
        if not isinstance(context, list) or len(context) != 2:
            continue
        title, sentences = context
        title = normalize_space(title)
        if isinstance(sentences, list):
            text = normalize_space(" ".join(str(sentence) for sentence in sentences))
        else:
            text = normalize_space(sentences)
        if not title or not text:
            continue
        candidates.append(
            {
                "local_pid": f"{qid}_ctx_{index:02d}",
                "title": title,
                "text": text,
                "content": f"{title}. {text}",
                "context_index": index,
                "is_gold_title": title in gold_title_set,
            }
        )
    return candidates


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a fixed 500-question evaluation set from HotpotQA dev distractor.")
    parser.add_argument(
        "--input",
        default="setr_reproduction/data/raw/hotpotqa/hotpot_dev_distractor_v1.json",
        help="Path to hotpot_dev_distractor_v1.json.",
    )
    parser.add_argument("--output-dir", default="setr_reproduction/data/processed/eval")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    valid_items = [item for item in dataset if item.get("_id") and item.get("question") and item.get("answer")]
    valid_items = [item for item in valid_items if extract_gold_titles(item)]
    if len(valid_items) < args.sample_size:
        raise ValueError(f"Not enough valid items: {len(valid_items)} < {args.sample_size}")

    rng = random.Random(args.seed)
    sampled_items = rng.sample(valid_items, args.sample_size)
    sampled_items = sorted(sampled_items, key=lambda item: item["_id"])

    full_rows = []
    query_rows = []
    candidate_rows = []
    qid_rows = []
    context_counts = []
    gold_title_counts = []
    gold_title_covered_counts = []
    level_counter = Counter()
    type_counter = Counter()

    for sample_index, item in enumerate(sampled_items):
        qid = item["_id"]
        gold_titles = extract_gold_titles(item)
        candidates = build_candidate_contexts(item, gold_titles)
        candidate_titles = {candidate["title"] for candidate in candidates}
        covered_gold_titles = [title for title in gold_titles if title in candidate_titles]

        base = {
            "sample_index": sample_index,
            "qid": qid,
            "question": normalize_space(item["question"]),
            "answer": normalize_space(item["answer"]),
            "level": item.get("level"),
            "type": item.get("type"),
            "supporting_facts": item.get("supporting_facts", []),
            "gold_titles": gold_titles,
        }
        full_rows.append({**base, "candidate_contexts": candidates})
        query_rows.append(base)
        candidate_rows.append(
            {
                "sample_index": sample_index,
                "qid": qid,
                "question": base["question"],
                "answer": base["answer"],
                "gold_titles": gold_titles,
                "candidate_contexts": candidates,
                "candidate_count": len(candidates),
                "gold_title_coverage_in_candidate_context": len(covered_gold_titles) / len(gold_titles),
                "all_gold_titles_in_candidate_context": set(gold_titles).issubset(candidate_titles),
            }
        )
        qid_rows.append({"sample_index": sample_index, "qid": qid})

        context_counts.append(len(candidates))
        gold_title_counts.append(len(gold_titles))
        gold_title_covered_counts.append(len(covered_gold_titles))
        level_counter[item.get("level")] += 1
        type_counter[item.get("type")] += 1

    hotpotqa_500_path = output_dir / "hotpotqa_500.jsonl"
    queries_path = output_dir / "queries_500.jsonl"
    candidate_context_path = output_dir / "candidate_context_500.jsonl"
    sampled_qids_path = output_dir / "sampled_qids_500.jsonl"
    manifest_path = output_dir / "sample_manifest.json"

    write_jsonl(hotpotqa_500_path, full_rows)
    write_jsonl(queries_path, query_rows)
    write_jsonl(candidate_context_path, candidate_rows)
    write_jsonl(sampled_qids_path, qid_rows)

    all_gold_titles_in_context = sum(
        1 for row in candidate_rows if row["all_gold_titles_in_candidate_context"]
    )
    manifest = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "source_dataset_items": len(dataset),
        "valid_items": len(valid_items),
        "outputs": {
            "hotpotqa_500": str(hotpotqa_500_path),
            "queries_500": str(queries_path),
            "candidate_context_500": str(candidate_context_path),
            "sampled_qids_500": str(sampled_qids_path),
        },
        "stats": {
            "samples": len(full_rows),
            "avg_candidate_contexts": sum(context_counts) / len(context_counts),
            "min_candidate_contexts": min(context_counts),
            "max_candidate_contexts": max(context_counts),
            "avg_gold_titles": sum(gold_title_counts) / len(gold_title_counts),
            "min_gold_titles": min(gold_title_counts),
            "max_gold_titles": max(gold_title_counts),
            "avg_gold_title_coverage_in_candidate_context": (
                sum(gold_title_covered_counts) / sum(gold_title_counts)
            ),
            "all_gold_titles_in_candidate_context_count": all_gold_titles_in_context,
            "all_gold_titles_in_candidate_context_rate": all_gold_titles_in_context / len(candidate_rows),
            "level_distribution": dict(level_counter),
            "type_distribution": dict(type_counter),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
