from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from watermark_lab.data import load_samples
from watermark_lab.metrics import binary_detection_metrics, perplexity
from watermark_lab.watermark import WatermarkConfig, build_logits_processor, detect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KGW/SWEET/EWD watermark comparison.")
    parser.add_argument("--config", default="configs/smoke_tiny.yaml", help="YAML experiment config.")
    parser.add_argument("--output-dir", default="results/smoke", help="Directory for CSV/JSONL outputs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override dataset sample count.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation length.")
    parser.add_argument("--skip-ppl", action="store_true", help="Skip oracle perplexity computation.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def dtype_from_config(name: str | None, device: torch.device):
    if not name or name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    return getattr(torch, name)


def load_model(model_cfg: dict, cache_dir: str, device: torch.device):
    kwargs = {
        "cache_dir": cache_dir,
        "torch_dtype": dtype_from_config(model_cfg.get("torch_dtype"), device),
        "local_files_only": bool(model_cfg.get("local_files_only", False)),
        "low_cpu_mem_usage": bool(model_cfg.get("low_cpu_mem_usage", True)),
    }
    if model_cfg.get("device_map"):
        kwargs["device_map"] = model_cfg["device_map"]
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    model.eval()
    return model


def load_tokenizer(model_cfg: dict | str, cache_dir: str):
    if isinstance(model_cfg, dict):
        model_name = model_cfg["name"]
        local_files_only = bool(model_cfg.get("local_files_only", False))
    else:
        model_name = model_cfg
        local_files_only = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LocalPPLScorer:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def score_text(self, prompt: str, completion: str) -> float:
        if not completion.strip():
            return float("nan")
        device = model_device(self.model)
        input_ids, prompt_length = encode_pair(self.tokenizer, prompt, completion, device)
        return perplexity(self.model, input_ids, prompt_length)

    def score_ids(self, input_ids: torch.Tensor, prompt_length: int) -> float:
        return perplexity(self.model, input_ids.to(model_device(self.model)), prompt_length)


def merged_watermark_config(base: dict, method: dict) -> WatermarkConfig:
    values = dict(base)
    for key in ("gamma", "delta", "key", "window_size", "entropy_threshold", "entropy_type", "spike_modulus"):
        if key in method:
            values[key] = method[key]
    values["detector"] = method["detector"]
    return WatermarkConfig(**values)


def encode_prompt(tokenizer, prompt: str, device: torch.device):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return encoded.to(device)


def encode_pair(tokenizer, prompt: str, completion: str, device: torch.device) -> tuple[torch.Tensor, int]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)
    return input_ids, int(prompt_ids.shape[1])


@torch.inference_mode()
def generate_completion(model, tokenizer, prompt: str, method: str, wm_cfg: WatermarkConfig, gen_cfg: dict):
    device = model_device(model)
    encoded = encode_prompt(tokenizer, prompt, device)
    input_ids = encoded.input_ids
    processor = build_logits_processor(method, wm_cfg, int(model.config.vocab_size))
    processors = None if processor is None else LogitsProcessorList([processor])
    max_new_tokens = int(gen_cfg["max_new_tokens"])
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=encoded.attention_mask,
        logits_processor=processors,
        max_new_tokens=max_new_tokens,
        do_sample=bool(gen_cfg.get("do_sample", True)),
        temperature=float(gen_cfg.get("temperature", 1.0)),
        top_p=float(gen_cfg.get("top_p", 1.0)),
        top_k=int(gen_cfg.get("top_k", 0)),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion_ids = output_ids[0, input_ids.shape[1] :]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion, output_ids.to(device), int(input_ids.shape[1])


def build_ppl_scorer(cfg: dict, args: argparse.Namespace, tokenizer, gen_model, cache_dir: str, device: torch.device):
    if args.skip_ppl:
        return None
    ppl_cfg = cfg.get("ppl", {})
    backend = ppl_cfg.get("backend", "local")
    if backend in {"none", "skip"}:
        return None
    if backend != "local":
        raise ValueError(f"Unsupported PPL backend: {backend}")

    oracle_cfg = cfg.get("oracle_model", cfg["generation_model"])
    oracle_model = gen_model
    if oracle_cfg["name"] != cfg["generation_model"]["name"]:
        oracle_model = load_model(oracle_cfg, cache_dir, device)
    return LocalPPLScorer(oracle_model, tokenizer)


def score_ppl_text(ppl_scorer, prompt: str, completion: str) -> float:
    if ppl_scorer is None:
        return float("nan")
    return ppl_scorer.score_text(prompt, completion)


def score_ppl_generated(ppl_scorer, input_ids: torch.Tensor, prompt_length: int, prompt: str, completion: str) -> float:
    if ppl_scorer is None:
        return float("nan")
    if hasattr(ppl_scorer, "score_ids"):
        return ppl_scorer.score_ids(input_ids, prompt_length)
    return ppl_scorer.score_text(prompt, completion)


def score_text(detector_model, tokenizer, prompt: str, completion: str, wm_cfg: WatermarkConfig) -> dict:
    device = model_device(detector_model)
    input_ids, prompt_length = encode_pair(tokenizer, prompt, completion, device)
    return detect(detector_model, input_ids, prompt_length, wm_cfg)


def score_ids(detector_model, input_ids: torch.Tensor, prompt_length: int, wm_cfg: WatermarkConfig) -> dict:
    return detect(detector_model, input_ids.to(model_device(detector_model)), prompt_length, wm_cfg)


def summarize(scores: pd.DataFrame, plain_ppl_mean: float) -> pd.DataFrame:
    rows = []
    for algorithm, group in scores.groupby("algorithm", sort=False):
        labels = group["label"].astype(int).tolist()
        z_scores = group["z_score"].astype(float).tolist()
        metrics = binary_detection_metrics(labels, z_scores)
        positives = group[group["label"] == 1]
        negatives = group[group["label"] == 0]
        pos_ppl = positives["ppl"].astype(float).replace([math.inf, -math.inf], np.nan)
        mean_ppl = float(pos_ppl.mean())
        rows.append(
            {
                "algorithm": algorithm,
                "n_positive": int(len(positives)),
                "n_negative": int(len(negatives)),
                "mean_z_positive": float(positives["z_score"].mean()),
                "mean_z_negative": float(negatives["z_score"].mean()),
                "mean_green_fraction_positive": float(positives["green_fraction"].mean()),
                "mean_scored_tokens_positive": float(positives["num_scored_tokens"].mean()),
                "mean_ppl_positive": mean_ppl,
                "ppl_delta_vs_plain": mean_ppl - plain_ppl_mean,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.max_new_tokens is not None:
        cfg["generation"]["max_new_tokens"] = args.max_new_tokens

    set_seed(int(cfg.get("seed", 0)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = cfg.get("model_cache_dir", "models/hf")
    device = choose_device(cfg.get("device", "auto"))
    tokenizer = load_tokenizer(cfg["generation_model"], cache_dir)
    gen_model = load_model(cfg["generation_model"], cache_dir, device)
    ppl_scorer = build_ppl_scorer(cfg, args, tokenizer, gen_model, cache_dir, device)

    samples = load_samples(cfg, tokenizer, args.max_samples)
    score_rows: list[dict] = []
    generation_rows: list[dict] = []
    plain_ppls: list[float] = []

    methods = cfg["methods"]
    for sample in tqdm(samples, desc="samples"):
        prompt = sample["prompt"]
        human_completion = sample["human_completion"]
        sample_generations: dict[str, tuple[str, float, torch.Tensor, int]] = {}
        plain_completion, plain_ids, plain_prompt_len = generate_completion(
            gen_model,
            tokenizer,
            prompt,
            "none",
            WatermarkConfig(**cfg["watermark"]),
            cfg["generation"],
        )
        plain_ppl = score_ppl_generated(ppl_scorer, plain_ids, plain_prompt_len, prompt, plain_completion)
        plain_ppls.append(plain_ppl)
        generation_rows.append(
            {
                "sample_id": sample.get("id"),
                "algorithm": "Plain",
                "prompt": prompt,
                "completion": plain_completion,
                "ppl": plain_ppl,
            }
        )

        for method in methods:
            wm_cfg = merged_watermark_config(cfg["watermark"], method)

            human_score = score_text(gen_model, tokenizer, prompt, human_completion, wm_cfg)
            human_ppl = score_ppl_text(ppl_scorer, prompt, human_completion)
            score_rows.append(
                {
                    "sample_id": sample.get("id"),
                    "algorithm": method["name"],
                    "source": "human",
                    "label": 0,
                    "completion": human_completion,
                    "ppl": human_ppl,
                    **human_score,
                }
            )

            reuse_name = method.get("reuse_generation_from")
            if reuse_name and reuse_name in sample_generations:
                completion, wm_ppl, completion_ids, completion_prompt_len = sample_generations[reuse_name]
            else:
                completion, completion_ids, completion_prompt_len = generate_completion(
                    gen_model,
                    tokenizer,
                    prompt,
                    method["generation"],
                    wm_cfg,
                    cfg["generation"],
                )
                wm_ppl = score_ppl_generated(ppl_scorer, completion_ids, completion_prompt_len, prompt, completion)
            sample_generations[method["name"]] = (completion, wm_ppl, completion_ids, completion_prompt_len)
            wm_score = score_ids(gen_model, completion_ids, completion_prompt_len, wm_cfg)
            score_rows.append(
                {
                    "sample_id": sample.get("id"),
                    "algorithm": method["name"],
                    "source": "watermarked",
                    "label": 1,
                    "completion": completion,
                    "ppl": wm_ppl,
                    **wm_score,
                }
            )
            generation_rows.append(
                {
                    "sample_id": sample.get("id"),
                    "algorithm": method["name"],
                    "prompt": prompt,
                    "completion": completion,
                    "ppl": wm_ppl,
                    "z_score": wm_score["z_score"],
                    "num_scored_tokens": wm_score["num_scored_tokens"],
                }
            )

    scores = pd.DataFrame(score_rows)
    plain_ppl_mean = float(pd.Series(plain_ppls, dtype=float).replace([math.inf, -math.inf], np.nan).mean())
    metrics = summarize(scores, plain_ppl_mean)

    scores.to_csv(output_dir / "scores.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as f:
        for row in generation_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(metrics.to_string(index=False))
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()
