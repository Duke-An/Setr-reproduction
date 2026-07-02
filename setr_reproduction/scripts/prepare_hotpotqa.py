import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def normalize_title(title: str) -> str:
    return title.lower().replace(" ", "_").replace("/", "_")


def convert_item(item: Dict) -> Dict:
    contexts = item.get("context", [])
    supporting_titles = {fact[0] for fact in item.get("supporting_facts", [])}
    candidates: List[Dict] = []
    supporting_passage_ids: List[str] = []

    item_id = item.get("_id") or item.get("id")
    for title, sentences in contexts:
        passage_id = f"{item_id}_{normalize_title(title)}"
        text = " ".join(sentences)
        candidates.append({
            "id": passage_id,
            "title": title,
            "text": text,
        })
        if title in supporting_titles:
            supporting_passage_ids.append(passage_id)

    return {
        "id": item_id,
        "question": item["question"],
        "answer": item["answer"],
        "supporting_passage_ids": supporting_passage_ids,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HotpotQA distractor-format JSON to local JSONL format.")
    parser.add_argument("--input", required=True, help="HotpotQA JSON file path.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples to keep.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    random.seed(args.seed)
    if args.sample_size > 0 and len(data) > args.sample_size:
        data = random.sample(data, args.sample_size)

    with output_path.open("w", encoding="utf-8") as file:
        for item in data:
            converted = convert_item(item)
            file.write(json.dumps(converted, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
