import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dedupe_key(title: str, text: str) -> str:
    normalized = f"{normalize_space(title).lower()}\n{normalize_space(text).lower()}"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def iter_contexts(item: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    for context in item.get("context", []):
        if not isinstance(context, list) or len(context) != 2:
            continue
        title, sentences = context
        if isinstance(sentences, list):
            text = " ".join(str(sentence) for sentence in sentences)
        else:
            text = str(sentences)
        title = normalize_space(str(title))
        text = normalize_space(text)
        if title and text:
            yield title, text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deduplicated paragraph-level corpus from HotpotQA dev distractor contexts."
    )
    parser.add_argument(
        "--input",
        default="setr_reproduction/data/raw/hotpotqa/hotpot_dev_distractor_v1.json",
        help="Path to hotpot_dev_distractor_v1.json.",
    )
    parser.add_argument(
        "--output",
        default="setr_reproduction/data/processed/corpus/hotpotqa_dev_distractor_corpus.jsonl",
        help="Output JSONL corpus path.",
    )
    parser.add_argument(
        "--stats",
        default="setr_reproduction/data/processed/corpus/hotpotqa_dev_distractor_corpus_stats.json",
        help="Output stats JSON path.",
    )
    parser.add_argument(
        "--title-index",
        default="setr_reproduction/data/processed/corpus/hotpotqa_dev_distractor_title_index.json",
        help="Output title to pid list mapping path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats)
    title_index_path = Path(args.title_index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    title_index_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    passage_map: Dict[str, Dict[str, Any]] = {}
    raw_context_count = 0
    context_count_per_question: List[int] = []
    title_counter: Counter[str] = Counter()

    for item in dataset:
        qid = item.get("_id", "")
        contexts = list(iter_contexts(item))
        context_count_per_question.append(len(contexts))
        for title, text in contexts:
            raw_context_count += 1
            title_counter[title] += 1
            key = dedupe_key(title, text)
            if key not in passage_map:
                passage_map[key] = {
                    "title": title,
                    "text": text,
                    "source_qids": [],
                }
            passage_map[key]["source_qids"].append(qid)

    sorted_items = sorted(
        passage_map.values(),
        key=lambda item: (item["title"].lower(), item["text"].lower()),
    )

    title_index: Dict[str, List[str]] = defaultdict(list)
    with output_path.open("w", encoding="utf-8") as file:
        for idx, item in enumerate(sorted_items):
            pid = f"hotpot_dev_distractor_{idx:06d}"
            source_qids = sorted(set(item["source_qids"]))
            record = {
                "pid": pid,
                "title": item["title"],
                "text": item["text"],
                "content": f"{item['title']}. {item['text']}",
                "source": "hotpotqa",
                "split": "dev_distractor",
                "granularity": "paragraph",
                "source_qids": source_qids,
                "source_count": len(item["source_qids"]),
            }
            title_index[item["title"]].append(pid)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    avg_contexts = sum(context_count_per_question) / max(len(context_count_per_question), 1)
    duplicate_context_count = raw_context_count - len(sorted_items)
    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dataset_items": len(dataset),
        "raw_context_count": raw_context_count,
        "deduplicated_passage_count": len(sorted_items),
        "duplicate_context_count": duplicate_context_count,
        "duplicate_ratio": duplicate_context_count / raw_context_count if raw_context_count else 0,
        "unique_title_count": len(title_counter),
        "avg_contexts_per_question": avg_contexts,
        "min_contexts_per_question": min(context_count_per_question) if context_count_per_question else 0,
        "max_contexts_per_question": max(context_count_per_question) if context_count_per_question else 0,
        "top_repeated_titles": title_counter.most_common(20),
        "dedupe_rule": "sha1(lower(normalize_space(title)) + newline + lower(normalize_space(joined_context_text))))",
        "content_rule": "content = title + '. ' + text",
    }
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    with title_index_path.open("w", encoding="utf-8") as file:
        json.dump(title_index, file, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
