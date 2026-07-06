from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml


MODEL_ALLOW_PATTERNS = [
    "*.bin",
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
    "merges.txt",
    "tokenizer.*",
    "vocab.*",
]

DEFAULT_HF_MIRROR = "https://hf-mirror.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download models and dataset caches into this project.")
    parser.add_argument("--config", default="configs/opt_full.yaml")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HF endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--hf-mirror", action="store_true", help="Shortcut for --hf-endpoint https://hf-mirror.com.")
    parser.add_argument("--no-mirror-fallback", action="store_true", help="Do not retry Hugging Face downloads via hf-mirror.com.")
    parser.add_argument("--xet-high-performance", action="store_true", help="Enable HF_XET_HIGH_PERFORMANCE=1.")
    parser.add_argument("--disable-xet", action="store_true", help="Disable hf-xet if it is unstable on this network.")
    parser.add_argument("--source", choices=["hf", "modelscope"], default="hf", help="Model download source.")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--dataset-samples", type=int, default=200)
    parser.add_argument(
        "--dataset-output",
        default="data/c4_realnewslike_200.jsonl",
        help="Local JSONL subset path for streaming dataset download.",
    )
    return parser.parse_args()


def effective_hf_endpoint(args: argparse.Namespace) -> str | None:
    if args.hf_endpoint:
        return args.hf_endpoint
    if args.hf_mirror:
        return DEFAULT_HF_MIRROR
    return os.environ.get("HF_ENDPOINT")


def configure_hf_env(args: argparse.Namespace, cfg: dict) -> str | None:
    cache_dir = Path(cfg.get("model_cache_dir", "models/hf"))
    os.environ.setdefault("HF_HOME", str((ROOT / "models" / "hf_home").resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((ROOT / cache_dir).resolve()))
    os.environ.setdefault("HF_ASSETS_CACHE", str((ROOT / "models" / "hf_assets").resolve()))
    endpoint = effective_hf_endpoint(args)
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    if args.xet_high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    print(f"HF endpoint: {endpoint or 'https://huggingface.co'}")
    print(f"HF cache: {os.environ['HF_HUB_CACHE']}")
    return endpoint


def download_hf_model(model_name: str, cache_dir: str, endpoint: str | None, mirror_fallback: bool) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_name} into HF cache {cache_dir}")
    try:
        snapshot_download(
            repo_id=model_name,
            repo_type="model",
            cache_dir=cache_dir,
            allow_patterns=MODEL_ALLOW_PATTERNS,
            endpoint=endpoint,
            resume_download=True,
        )
    except Exception as exc:
        if endpoint or not mirror_fallback:
            raise RuntimeError(
                "Hugging Face download failed. If the error host is huggingface.co, "
                "retry with: --hf-mirror"
            ) from exc
        print(f"Download from huggingface.co failed: {exc}")
        print(f"Retrying {model_name} via {DEFAULT_HF_MIRROR}")
        os.environ["HF_ENDPOINT"] = DEFAULT_HF_MIRROR
        snapshot_download(
            repo_id=model_name,
            repo_type="model",
            cache_dir=cache_dir,
            allow_patterns=MODEL_ALLOW_PATTERNS,
            endpoint=DEFAULT_HF_MIRROR,
            resume_download=True,
        )


def download_modelscope_model(model_cfg: dict, cache_dir: str) -> None:
    from modelscope import snapshot_download

    model_id = model_cfg.get("modelscope_id", model_cfg["name"])
    local_dir = Path(cache_dir) / model_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} from ModelScope into {local_dir}")
    snapshot_download(model_id=model_id, local_dir=str(local_dir))


def modelscope_local_dir(model_cfg: dict, cache_dir: str) -> str:
    model_id = model_cfg.get("modelscope_id", model_cfg["name"])
    return str(Path(cache_dir) / model_id.replace("/", "__"))


def build_tokenizer(model_name: str, cache_dir: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def write_streaming_dataset_subset(cfg: dict, tokenizer, output_path: str, sample_count: int) -> None:
    from datasets import load_dataset

    data_cfg = cfg["dataset"]
    output = ROOT / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    needed = int(data_cfg.get("prompt_tokens", 64)) + int(data_cfg.get("completion_tokens", 200))
    print(f"Streaming {sample_count} usable samples from {data_cfg['name']} into {output}")
    dataset = load_dataset(
        data_cfg["name"],
        data_cfg.get("config"),
        split=data_cfg.get("split", "validation"),
        cache_dir=data_cfg.get("cache_dir", "data/hf_datasets"),
        streaming=True,
    )

    written = 0
    with output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(dataset):
            text = row.get("text") or row.get("content") or ""
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < needed:
                continue
            prompt_ids = token_ids[: int(data_cfg.get("prompt_tokens", 64))]
            completion_ids = token_ids[
                int(data_cfg.get("prompt_tokens", 64)) : needed
            ]
            f.write(
                json.dumps(
                    {
                        "id": str(row.get("id", written)),
                        "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                        "human_completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if written >= sample_count:
                break
    if written == 0:
        raise RuntimeError("No usable samples were written. Check dataset access and split/config names.")
    print(f"Wrote {written} samples to {output}")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    endpoint = configure_hf_env(args, cfg)
    cache_dir = cfg.get("model_cache_dir", "models/hf")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if not args.skip_models:
        for section in ["generation_model", "oracle_model"]:
            if args.source == "hf":
                download_hf_model(
                    cfg[section]["name"],
                    cache_dir,
                    endpoint,
                    mirror_fallback=not args.no_mirror_fallback,
                )
            else:
                download_modelscope_model(cfg[section], cache_dir)

    if not args.skip_dataset and cfg["dataset"]["source"] == "hf_text":
        tokenizer_name = cfg["generation_model"]["name"]
        if args.source == "modelscope":
            tokenizer_name = modelscope_local_dir(cfg["generation_model"], cache_dir)
        tokenizer = build_tokenizer(tokenizer_name, cache_dir)
        write_streaming_dataset_subset(cfg, tokenizer, args.dataset_output, args.dataset_samples)


if __name__ == "__main__":
    main()
