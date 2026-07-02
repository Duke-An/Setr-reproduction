import argparse
import urllib.request
from pathlib import Path


URLS = [
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json",
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HotpotQA distractor dev JSON.")
    parser.add_argument(
        "--output",
        default="data/raw/hotpot_dev_distractor_v1.json",
        help="Output path under setr_reproduction or an absolute path.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = None
    for url in URLS:
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, output_path)
            print(f"Saved to {output_path}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Failed: {exc}")

    raise RuntimeError(f"All HotpotQA download URLs failed. Last error: {last_error}")


if __name__ == "__main__":
    main()
