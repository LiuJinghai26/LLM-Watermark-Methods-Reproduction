from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from scripts.run_experiment import (
    WatermarkConfig,
    choose_device,
    generate_completion,
    load_model,
    load_tokenizer,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether OPT-1.3B can load and generate locally.")
    parser.add_argument("--config", default="configs/opt_self_ppl.yaml")
    parser.add_argument("--prompt", default="The watermark detection experiment studies")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--watermark", choices=["none", "kgw", "sweet"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["generation"]["max_new_tokens"] = args.max_new_tokens
    set_seed(int(cfg.get("seed", 0)))

    device = choose_device(cfg.get("device", "auto"))
    cache_dir = cfg.get("model_cache_dir", "models/hf")
    print(f"Loading {cfg['generation_model']['name']} on {device}...")
    tokenizer = load_tokenizer(cfg["generation_model"], cache_dir)
    model = load_model(cfg["generation_model"], cache_dir, device)
    print("Model loaded.")
    if torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        print(f"CUDA memory allocated/reserved: {used_gb:.2f} GB / {reserved_gb:.2f} GB")

    method = "none" if args.watermark == "none" else args.watermark
    completion, output_ids, prompt_len = generate_completion(
        model,
        tokenizer,
        args.prompt,
        method,
        WatermarkConfig(**cfg["watermark"]),
        cfg["generation"],
    )
    print(f"Prompt tokens: {prompt_len}")
    print(f"Output tokens: {int(output_ids.shape[1] - prompt_len)}")
    print("Completion:")
    print(completion)


if __name__ == "__main__":
    main()
