from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a HumanEval JSONL dataset for watermark experiments.")
    parser.add_argument("--output", default="data/humaneval_164.jsonl")
    parser.add_argument("--cache-dir", default="data/hf_datasets")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=None, help="Optional limit for quick subsets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: install with `python -m pip install datasets`.") from exc

    dataset = load_dataset(
        "openai/openai_humaneval",
        split=args.split,
        cache_dir=args.cache_dir,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(
                json.dumps(
                    {
                        "id": row["task_id"],
                        "prompt": row["prompt"],
                        "human_completion": row["canonical_solution"],
                        "entry_point": row.get("entry_point"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if args.samples is not None and written >= args.samples:
                break

    print(f"Wrote {written} HumanEval samples to {output}")


if __name__ == "__main__":
    main()
