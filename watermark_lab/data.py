from __future__ import annotations

import json
from pathlib import Path


def load_local_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def load_hf_text_dataset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    cache_dir: str | Path,
    tokenizer,
    prompt_tokens: int,
    completion_tokens: int,
    max_samples: int,
) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        cache_dir=str(cache_dir),
        streaming=False,
    )

    samples: list[dict] = []
    needed = prompt_tokens + completion_tokens
    for row in dataset:
        text = row.get("text") or row.get("content") or ""
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < needed:
            continue
        prompt_ids = token_ids[:prompt_tokens]
        completion_ids = token_ids[prompt_tokens:needed]
        samples.append(
            {
                "id": str(row.get("id", len(samples))),
                "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                "human_completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
            }
        )
        if len(samples) >= max_samples:
            break

    if len(samples) < max_samples:
        raise RuntimeError(
            f"Only found {len(samples)} usable samples with at least {needed} tokens in {dataset_name}."
        )
    return samples


def load_samples(cfg: dict, tokenizer, max_samples: int | None = None) -> list[dict]:
    data_cfg = cfg["dataset"]
    sample_count = int(max_samples or data_cfg.get("max_samples", 100))
    source = data_cfg.get("source", "local_jsonl")
    if source == "local_jsonl":
        return load_local_jsonl(data_cfg["path"], sample_count)
    if source == "hf_text":
        return load_hf_text_dataset(
            dataset_name=data_cfg["name"],
            dataset_config=data_cfg.get("config"),
            split=data_cfg.get("split", "validation"),
            cache_dir=data_cfg.get("cache_dir", "data/hf_datasets"),
            tokenizer=tokenizer,
            prompt_tokens=int(data_cfg.get("prompt_tokens", 64)),
            completion_tokens=int(data_cfg.get("completion_tokens", 128)),
            max_samples=sample_count,
        )
    raise ValueError(f"Unsupported dataset source: {source}")
