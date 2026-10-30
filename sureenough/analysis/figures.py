from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAIVE, OURS = "#c2410c", "#1d4ed8"


def _promised_vs_delivered(per_subject: dict, alpha: float, path: Path) -> None:
    subjects = sorted(per_subject, key=int)
    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - 0.2, [per_subject[s]["naive_pop_risk"] for s in subjects], 0.4, color=NAIVE, label="Check after every repeat")
    ax.bar(x + 0.2, [per_subject[s]["pop_risk"] for s in subjects], 0.4, color=OURS, label="SureEnough")
    ax.axhline(alpha, color="black", lw=1.2, ls="--", label=f"Promise: at most {alpha:.0%} wrong")
    ax.set_xticks(x, [f"S{s}" for s in subjects])
    ax.set_ylabel("Share of images missed")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fixed_k_hull(per_subject: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for s, summary in sorted(per_subject.items(), key=lambda kv: int(kv[0])):
        sizes, repeats = zip(*summary["fixed_points"])
        ax.plot(sizes, repeats, color="0.7", lw=1)
        ax.scatter([summary["mean_set_size"]], [summary["mean_stop"]], color=OURS, s=18, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Mean final answer size")
    ax.set_ylabel("Mean repeats used")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _stop_histogram(per_subject: dict, path: Path) -> None:
    stops = np.concatenate([np.asarray(s["per_image"]["stop"]) for s in per_subject.values()])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(stops, bins=np.arange(0.5, 21.5, 1), color=OURS)
    ax.set_xlabel("Repeats before stopping")
    ax.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_figures(results: dict, alpha: float, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subject = results["per_subject"]
    paths = [out_dir / "promised_vs_delivered.png", out_dir / "fixed_k_hull.png", out_dir / "stop_histogram.png"]
    _promised_vs_delivered(per_subject, alpha, paths[0])
    _fixed_k_hull(per_subject, paths[1])
    _stop_histogram(per_subject, paths[2])
    return paths
