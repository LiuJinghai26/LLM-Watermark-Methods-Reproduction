from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve


@torch.inference_mode()
def perplexity(model, input_ids: torch.Tensor, prompt_length: int) -> float:
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    outputs = model(input_ids=input_ids, labels=labels)
    loss = float(outputs.loss.detach().cpu().item())
    return float(math.exp(loss)) if math.isfinite(loss) else float("nan")


def binary_detection_metrics(labels: list[int], scores: list[float]) -> dict:
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores, dtype=float)
    if len(set(labels_arr.tolist())) < 2:
        return {"auroc": float("nan"), "tpr_at_5pct_fpr": float("nan")}

    fpr, tpr, _thresholds = roc_curve(labels_arr, scores_arr)
    auroc = float(auc(fpr, tpr))
    allowed = np.where(fpr <= 0.05)[0]
    tpr_at_5 = float(np.max(tpr[allowed])) if len(allowed) else 0.0
    return {"auroc": auroc, "tpr_at_5pct_fpr": tpr_at_5}
