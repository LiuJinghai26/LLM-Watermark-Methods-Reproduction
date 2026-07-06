from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import LogitsProcessor


@dataclass(frozen=True)
class WatermarkConfig:
    gamma: float = 0.25
    delta: float = 2.0
    key: int = 15485863
    window_size: int = 1
    entropy_threshold: float = 1.0
    entropy_type: str = "shannon"
    spike_modulus: float | None = None
    detector: str = "kgw"


def normal_survival(z_score: float) -> float:
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def spike_modulus_from_params(gamma: float, delta: float) -> float:
    delta_exp = math.exp(delta) - 1.0
    return ((1.0 - gamma) * delta_exp) / (1.0 + gamma * delta_exp)


def stable_seed(context_tokens: Iterable[int], key: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(int(key).to_bytes(8, "little", signed=False))
    for token in context_tokens:
        h.update(int(token).to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest(), "big", signed=False)


def greenlist_ids(
    context_tokens: torch.Tensor | list[int],
    vocab_size: int,
    gamma: float,
    key: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    green_count = max(1, min(vocab_size - 1, int(round(gamma * vocab_size))))
    if isinstance(context_tokens, torch.Tensor):
        context = context_tokens.detach().cpu().tolist()
    else:
        context = context_tokens
    seed = stable_seed(context, key) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(vocab_size, generator=generator, device="cpu")[:green_count].to(device)


def is_green_token(
    token_id: int,
    context_tokens: torch.Tensor | list[int],
    vocab_size: int,
    gamma: float,
    key: int,
) -> bool:
    ids = greenlist_ids(context_tokens, vocab_size, gamma, key, device="cpu")
    return bool((ids == int(token_id)).any().item())


def shannon_entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-30))
    return float(-(probs * log_probs).sum().item())


def spike_entropy_from_logits(logits: torch.Tensor, modulus: float) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    return float((probs / (1.0 + modulus * probs)).sum().item())


def entropy_from_logits(logits: torch.Tensor, cfg: WatermarkConfig) -> float:
    if cfg.entropy_type == "spike":
        modulus = cfg.spike_modulus or spike_modulus_from_params(cfg.gamma, cfg.delta)
        return spike_entropy_from_logits(logits, modulus)
    if cfg.entropy_type == "shannon":
        return shannon_entropy_from_logits(logits)
    raise ValueError(f"Unsupported entropy_type: {cfg.entropy_type}")


class KGWLogitsProcessor(LogitsProcessor):
    def __init__(self, cfg: WatermarkConfig, vocab_size: int):
        self.cfg = cfg
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.clone()
        for row_idx in range(input_ids.shape[0]):
            context = input_ids[row_idx, -self.cfg.window_size :]
            green_ids = greenlist_ids(
                context,
                self.vocab_size,
                self.cfg.gamma,
                self.cfg.key,
                device=scores.device,
            )
            scores[row_idx, green_ids] += self.cfg.delta
        return scores


class SWEETLogitsProcessor(KGWLogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.clone()
        for row_idx in range(input_ids.shape[0]):
            entropy = entropy_from_logits(scores[row_idx], self.cfg)
            if entropy <= self.cfg.entropy_threshold:
                continue
            context = input_ids[row_idx, -self.cfg.window_size :]
            green_ids = greenlist_ids(
                context,
                self.vocab_size,
                self.cfg.gamma,
                self.cfg.key,
                device=scores.device,
            )
            scores[row_idx, green_ids] += self.cfg.delta
        return scores


def build_logits_processor(method: str, cfg: WatermarkConfig, vocab_size: int) -> LogitsProcessor | None:
    method = method.lower()
    if method in {"none", "plain", "unwatermarked"}:
        return None
    if method == "kgw":
        return KGWLogitsProcessor(cfg, vocab_size)
    if method == "sweet":
        return SWEETLogitsProcessor(cfg, vocab_size)
    if method == "ewd":
        return KGWLogitsProcessor(cfg, vocab_size)
    raise ValueError(f"Unsupported watermark generation method: {method}")


@torch.inference_mode()
def detect(
    model,
    input_ids: torch.Tensor,
    prompt_length: int,
    cfg: WatermarkConfig,
) -> dict:
    detector = cfg.detector.lower()
    token_ids = input_ids.detach().cpu()[0].tolist()
    vocab_size = int(getattr(model.config, "vocab_size"))
    start = max(1, int(prompt_length))
    end = len(token_ids)

    logits = None
    if detector in {"sweet", "ewd"}:
        outputs = model(input_ids=input_ids)
        logits = outputs.logits.detach().cpu()[0]

    green_count = 0
    green_weight = 0.0
    weight_sum = 0.0
    weight_sq_sum = 0.0
    scored = 0
    entropies: list[float] = []

    ewd_modulus = cfg.spike_modulus or spike_modulus_from_params(cfg.gamma, cfg.delta)
    ewd_min_entropy = 1.0 / (1.0 + ewd_modulus)

    for pos in range(start, end):
        context = token_ids[max(0, pos - cfg.window_size) : pos]
        token_id = token_ids[pos]
        entropy = None
        if logits is not None:
            entropy = entropy_from_logits(logits[pos - 1], cfg)

        if detector == "kgw":
            weight = 1.0
        elif detector == "sweet":
            if entropy is None or entropy <= cfg.entropy_threshold:
                continue
            weight = 1.0
        elif detector == "ewd":
            if entropy is None:
                continue
            if cfg.entropy_type == "spike":
                raw_weight = entropy - ewd_min_entropy
            else:
                raw_weight = entropy
            weight = max(0.0, float(raw_weight))
            if weight <= 0.0:
                continue
        else:
            raise ValueError(f"Unsupported detector: {cfg.detector}")

        is_green = is_green_token(token_id, context, vocab_size, cfg.gamma, cfg.key)
        scored += 1
        weight_sum += weight
        weight_sq_sum += weight * weight
        if entropy is not None:
            entropies.append(entropy)
        if is_green:
            green_count += 1
            green_weight += weight

    denom = math.sqrt(cfg.gamma * (1.0 - cfg.gamma) * weight_sq_sum)
    z_score = 0.0 if denom == 0.0 else (green_weight - cfg.gamma * weight_sum) / denom
    p_value = normal_survival(z_score)
    green_fraction = 0.0 if scored == 0 else green_count / scored
    entropy_mean = float("nan") if not entropies else sum(entropies) / len(entropies)

    return {
        "detector": detector,
        "z_score": z_score,
        "p_value": p_value,
        "num_scored_tokens": scored,
        "green_count": green_count,
        "green_fraction": green_fraction,
        "green_weight": green_weight,
        "weight_sum": weight_sum,
        "weight_sq_sum": weight_sq_sum,
        "entropy_mean": entropy_mean,
    }
