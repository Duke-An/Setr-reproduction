import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile(sorted_values: List[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def length_summary(values: List[int]) -> Dict[str, float]:
    values_sorted = sorted(values)
    return {
        "min": min(values_sorted) if values_sorted else 0,
        "p25": percentile(values_sorted, 0.25),
        "median": median(values_sorted) if values_sorted else 0,
        "mean": mean(values_sorted) if values_sorted else 0,
        "p75": percentile(values_sorted, 0.75),
        "p90": percentile(values_sorted, 0.90),
        "p95": percentile(values_sorted, 0.95),
        "p99": percentile(values_sorted, 0.99),
        "max": max(values_sorted) if values_sorted else 0,
    }


def short_record(record: Dict[str, Any], word_count: int, char_count: int) -> Dict[str, Any]:
    return {
        "pid": record["pid"],
        "title": record["title"],
        "word_count": word_count,
        "char_count": char_count,
        "source_count": record.get("source_count", 0),
        "text_preview": record["text"][:300],
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HotpotQA dev distractor corpus summary",
        "",
        "## 基本统计",
        "",
        f"- Corpus 文件：`{summary['input_path']}`",
        f"- Passage 数：{summary['passage_count']}",
        f"- 唯一 PID 数：{summary['unique_pid_count']}",
        f"- 唯一 Title 数：{summary['unique_title_count']}",
        f"- Source：{summary['source_distribution']}",
        f"- Split：{summary['split_distribution']}",
        "",
        "## 长度统计",
        "",
        "| 指标 | 词数 | 字符数 |",
        "|---|---:|---:|",
    ]
    for key in ["min", "p25", "median", "mean", "p75", "p90", "p95", "p99", "max"]:
        lines.append(
            f"| {key} | {summary['word_count_summary'][key]:.2f} | {summary['char_count_summary'][key]:.2f} |"
        )

    lines += [
        "",
        "## 阈值统计",
        "",
        "| 条件 | 数量 |",
        "|---|---:|",
    ]
    for key, value in summary["threshold_counts"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## source_count 分布",
        "",
        "| source_count | passage 数 |",
        "|---:|---:|",
    ]
    for key, value in summary["source_count_distribution"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## 最长段落 Top 10",
        "",
        "| rank | pid | title | word_count | source_count |",
        "|---:|---|---|---:|---:|",
    ]
    for rank, item in enumerate(summary["longest_passages"], start=1):
        lines.append(
            f"| {rank} | `{item['pid']}` | {item['title']} | {item['word_count']} | {item['source_count']} |"
        )

    lines += [
        "",
        "## 重复出现最多的 passage Top 10",
        "",
        "| rank | pid | title | source_count | word_count |",
        "|---:|---|---|---:|---:|",
    ]
    for rank, item in enumerate(summary["most_reused_passages"], start=1):
        lines.append(
            f"| {rank} | `{item['pid']}` | {item['title']} | {item['source_count']} | {item['word_count']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize HotpotQA dev distractor corpus JSONL.")
    parser.add_argument(
        "--input",
        default="setr_reproduction/data/processed/corpus/hotpotqa_dev_distractor_corpus.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default="setr_reproduction/data/processed/corpus/hotpotqa_dev_distractor_corpus_summary.json",
    )
    parser.add_argument(
        "--output-md",
        default="setr_reproduction/data/processed/corpus/hotpotqa_dev_distractor_corpus_summary.md",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    records = []
    pid_counter = Counter()
    title_counter = Counter()
    source_counter = Counter()
    split_counter = Counter()
    source_count_counter = Counter()
    word_counts = []
    char_counts = []

    for record in read_jsonl(input_path):
        text = record["text"]
        word_count = len(WORD_RE.findall(text))
        char_count = len(text)
        records.append((record, word_count, char_count))
        pid_counter[record["pid"]] += 1
        title_counter[record["title"]] += 1
        source_counter[record.get("source", "")] += 1
        split_counter[record.get("split", "")] += 1
        source_count_counter[str(record.get("source_count", 0))] += 1
        word_counts.append(word_count)
        char_counts.append(char_count)

    longest = sorted(records, key=lambda item: item[1], reverse=True)[:10]
    most_reused = sorted(records, key=lambda item: item[0].get("source_count", 0), reverse=True)[:10]

    summary = {
        "input_path": str(input_path),
        "passage_count": len(records),
        "unique_pid_count": len(pid_counter),
        "duplicate_pid_count": sum(count - 1 for count in pid_counter.values() if count > 1),
        "unique_title_count": len(title_counter),
        "titles_with_multiple_passages": sum(1 for count in title_counter.values() if count > 1),
        "source_distribution": dict(source_counter),
        "split_distribution": dict(split_counter),
        "word_count_summary": length_summary(word_counts),
        "char_count_summary": length_summary(char_counts),
        "threshold_counts": {
            "word_count_gt_512": sum(1 for value in word_counts if value > 512),
            "word_count_gt_1000": sum(1 for value in word_counts if value > 1000),
            "word_count_gt_1500": sum(1 for value in word_counts if value > 1500),
            "word_count_gt_2500": sum(1 for value in word_counts if value > 2500),
            "char_count_gt_4000": sum(1 for value in char_counts if value > 4000),
            "char_count_gt_8000": sum(1 for value in char_counts if value > 8000),
        },
        "source_count_distribution": dict(sorted(source_count_counter.items(), key=lambda item: int(item[0]))),
        "longest_passages": [short_record(record, wc, cc) for record, wc, cc in longest],
        "most_reused_passages": [short_record(record, wc, cc) for record, wc, cc in most_reused],
    }

    write_json(Path(args.output_json), summary)
    write_markdown(Path(args.output_md), summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2)[:6000])
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
