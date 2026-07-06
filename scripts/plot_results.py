from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot watermark experiment results.")
    parser.add_argument("--results-dir", default="results/smoke")
    parser.add_argument("--figures-dir", default=None)
    return parser.parse_args()


def save_z_scores(scores: pd.DataFrame, figures_dir: Path) -> None:
    algorithms = list(scores["algorithm"].drop_duplicates())
    data = []
    labels = []
    positions = []
    pos = 1
    for algorithm in algorithms:
        for source in ["human", "watermarked"]:
            subset = scores[(scores["algorithm"] == algorithm) & (scores["source"] == source)]
            data.append(subset["z_score"].astype(float).tolist())
            labels.append(f"{algorithm}\n{source}")
            positions.append(pos)
            pos += 1
        pos += 0.5

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(data, positions=positions, widths=0.55, showfliers=True)
    ax.axhline(4.0, color="#b02a37", linestyle="--", linewidth=1.2, label="z=4")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Detection z-score")
    ax.set_title("Watermark Detection Score Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "z_scores.png", dpi=220)
    plt.close(fig)


def save_roc(scores: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    for algorithm, group in scores.groupby("algorithm", sort=False):
        if group["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(group["label"].astype(int), group["z_score"].astype(float))
        ax.plot(fpr, tpr, marker="o", linewidth=1.6, label=algorithm)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png", dpi=220)
    plt.close(fig)


def save_tradeoff(metrics: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    colors = ["#1f7a8c", "#b56576", "#4d908e", "#e09f3e"]
    offsets = [(8, 10), (8, -16), (-38, 10), (-38, -16)]
    for idx, row in metrics.reset_index(drop=True).iterrows():
        color = colors[idx % len(colors)]
        ax.scatter(row["mean_ppl_positive"], row["mean_z_positive"], s=88, color=color)
        ax.annotate(
            row["algorithm"],
            (row["mean_ppl_positive"], row["mean_z_positive"]),
            xytext=offsets[idx % len(offsets)],
            textcoords="offset points",
        )
    ax.set_xlabel("Mean PPL of Generated Text")
    ax.set_ylabel("Mean Positive z-score")
    ax.set_title("Quality-Detectability Tradeoff")
    ax.grid(alpha=0.25)
    ax.margins(x=0.06, y=0.14)
    fig.tight_layout()
    fig.savefig(figures_dir / "quality_tradeoff.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(results_dir / "scores.csv")
    metrics = pd.read_csv(results_dir / "metrics.csv")
    save_z_scores(scores, figures_dir)
    save_roc(scores, figures_dir)
    save_tradeoff(metrics, figures_dir)
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
