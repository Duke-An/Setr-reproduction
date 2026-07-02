import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sample corpus passages as plain text for Dify testing.")
    parser.add_argument(
        "--input",
        default="setr_reproduction/data/processed/corpus/hotpotqa_dev_distractor_corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        default="setr_reproduction/data/processed/corpus/dify_test_10_passages.txt",
    )
    parser.add_argument(
        "--metadata-output",
        default="setr_reproduction/data/processed/corpus/dify_test_10_metadata.jsonl",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to export. Use 0 or negative for all rows.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_output_path = Path(args.metadata_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    metadata_rows = []
    with input_path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if args.limit > 0 and index >= args.limit:
                break
            row = json.loads(line)
            chunks.append(
                "\n".join(
                    [
                        f"PID: {row['pid']}",
                        f"Title: {row['title']}",
                        f"Content: {row['text']}",
                    ]
                )
            )
            metadata_rows.append(
                {
                    "content": row["content"],
                    "metadata": {
                        "pid": row["pid"],
                        "title": row["title"],
                        "source": row["source"],
                        "split": row["split"],
                        "granularity": row["granularity"],
                        "source_count": row["source_count"],
                    },
                }
            )

    output_path.write_text("\n=====\n".join(chunks), encoding="utf-8")
    with metadata_output_path.open("w", encoding="utf-8") as file:
        for row in metadata_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(output_path)
    print(output_path.stat().st_size)
    print(metadata_output_path)
    print(metadata_output_path.stat().st_size)


if __name__ == "__main__":
    main()
