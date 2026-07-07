from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = [
    (
        "return the sum of all even values in nums",
        "nums",
        [
            "    total = 0",
            "    for value in nums:",
            "        if value % 2 == 0:",
            "            total += value",
            "    return total",
        ],
    ),
    (
        "count how many strings in items are not empty",
        "items",
        [
            "    count = 0",
            "    for item in items:",
            "        if item:",
            "            count += 1",
            "    return count",
        ],
    ),
    (
        "return a list with every negative number replaced by zero",
        "values",
        [
            "    cleaned = []",
            "    for value in values:",
            "        if value < 0:",
            "            cleaned.append(0)",
            "        else:",
            "            cleaned.append(value)",
            "    return cleaned",
        ],
    ),
    (
        "return True when text is a palindrome after lowercasing",
        "text",
        [
            "    normalized = text.lower()",
            "    return normalized == normalized[::-1]",
        ],
    ),
    (
        "build a dictionary that maps each word to its length",
        "words",
        [
            "    result = {}",
            "    for word in words:",
            "        result[word] = len(word)",
            "    return result",
        ],
    ),
    (
        "return the maximum value or None when values is empty",
        "values",
        [
            "    if not values:",
            "        return None",
            "    best = values[0]",
            "    for value in values[1:]:",
            "        if value > best:",
            "            best = value",
            "    return best",
        ],
    ),
    (
        "count the number of vowels in text",
        "text",
        [
            "    vowels = set(\"aeiouAEIOU\")",
            "    total = 0",
            "    for char in text:",
            "        if char in vowels:",
            "            total += 1",
            "    return total",
        ],
    ),
    (
        "return a sorted list of the unique values",
        "values",
        [
            "    seen = set(values)",
            "    return sorted(seen)",
        ],
    ),
    (
        "return the first item that appears twice, otherwise None",
        "items",
        [
            "    seen = set()",
            "    for item in items:",
            "        if item in seen:",
            "            return item",
            "        seen.add(item)",
            "    return None",
        ],
    ),
    (
        "clamp value so it stays between lower and upper",
        "value, lower, upper",
        [
            "    if value < lower:",
            "        return lower",
            "    if value > upper:",
            "        return upper",
            "    return value",
        ],
    ),
]


def build_sample(index: int) -> dict[str, str]:
    description, args, body = TASKS[index % len(TASKS)]
    name = f"task_{index:03d}"
    prompt = (
        f"# Task {index:03d}: {description}.\n"
        f"# Keep the implementation short and deterministic.\n"
        f"def {name}({args}):\n"
    )
    completion = "\n".join(body) + "\n"
    return {
        "id": f"code-{index:03d}",
        "prompt": prompt,
        "human_completion": completion,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a low-entropy Python code JSONL dataset.")
    parser.add_argument("--output", default="data/code_low_entropy_200.jsonl")
    parser.add_argument("--samples", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for index in range(args.samples):
            f.write(json.dumps(build_sample(index), ensure_ascii=True) + "\n")
    print(f"Wrote {args.samples} samples to {output}")


if __name__ == "__main__":
    main()
