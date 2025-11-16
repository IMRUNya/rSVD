"""Synthetic randomized SVD benchmarks on ~10^6-entry matrices.

The script generates multiple low-rank noisy matrices, runs our custom
truncated SVD and randomized SVD implementations, and reports the mean
relative Frobenius error plus runtime across all trials. Figures and
LaTeX tables mirror the workflow from rsvd_image_compression.ipynb.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------- configuration ---------------------------- #

NUM_MATRICES = 50
MATRIX_SHAPES: Tuple[Tuple[int, int], ...] = (
    (1000, 1000),
    (1200, 800),
    (800, 1200),
)
INTRINSIC_RANK_RANGE = (20, 200)
NOISE_RANGE = (0.01, 0.08)
DECAY = 0.92

RANKS_TO_TRY = [20, 40, 80, 120, 160, 200]
POWER_ITERATIONS = [0, 1, 2, 3, 4]
OVERSAMPLING = 15
RANDOM_STATE = 2024
BASE_SEED = 5150

FIGURE_DIR = Path("figures/synthetic_matrices")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------ data model ------------------------------ #

@dataclass
class MatrixSpec:
    name: str
    shape: Tuple[int, int]
    intrinsic_rank: int
    noise: float
    seed: int


# ------------------------- helper implementations ----------------------- #

def generate_specs(num_specs: int) -> List[MatrixSpec]:
    rng = np.random.default_rng(BASE_SEED)
    specs: List[MatrixSpec] = []
    for idx in range(num_specs):
        shape = MATRIX_SHAPES[idx % len(MATRIX_SHAPES)]
        intrinsic_rank = int(
            rng.integers(INTRINSIC_RANK_RANGE[0], INTRINSIC_RANK_RANGE[1] + 1)
        )
        noise = float(rng.uniform(NOISE_RANGE[0], NOISE_RANGE[1]))
        seed = int(rng.integers(0, 1_000_000))
        specs.append(
            MatrixSpec(
                name=f"Matrix-{idx:03d}",
                shape=shape,
                intrinsic_rank=intrinsic_rank,
                noise=noise,
                seed=seed,
            )
        )
    return specs


def make_low_rank_matrix(spec: MatrixSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    m, n = spec.shape
    k = min(spec.intrinsic_rank, m, n)
    left = rng.standard_normal((m, k))
    right = rng.standard_normal((k, n))
    singulars = DECAY ** np.arange(k)
    low_rank = (left * singulars) @ right
    noise = spec.noise * rng.standard_normal((m, n))
    return low_rank + noise


def truncated_svd(matrix: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    start = perf_counter()
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    elapsed = perf_counter() - start
    return (U, S, Vt), elapsed


def randomized_svd(
    matrix: np.ndarray,
    rank: int,
    oversampling: int,
    n_iter: int,
    random_state: int | None = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    m, n = matrix.shape
    k = min(rank, m, n)
    ell = min(k + oversampling, n)
    rng = np.random.default_rng(random_state)
    projector = rng.standard_normal(size=(n, ell))
    sample = matrix @ projector
    for _ in range(max(n_iter, 0)):
        sample = matrix @ (matrix.T @ sample)
    Q, _ = np.linalg.qr(sample, mode="reduced")
    B = Q.T @ matrix
    start = perf_counter()
    Uh, S, Vt = np.linalg.svd(B, full_matrices=False)
    elapsed = perf_counter() - start
    U = Q @ Uh
    return (U[:, :k], S[:k], Vt[:k, :]), elapsed


def reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, rank: int) -> np.ndarray:
    r = min(rank, len(S))
    return (U[:, :r] * S[:r]) @ Vt[:r, :]


def relative_frobenius_error(original: np.ndarray, approx: np.ndarray) -> float:
    denom = np.linalg.norm(original, ord="fro")
    if denom == 0:
        return 0.0
    return np.linalg.norm(original - approx, ord="fro") / denom


def sanitize_filename(text: str, default: str = "figure") -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)
    cleaned = cleaned.strip("_")[:120]
    return cleaned or default


# ------------------------------ experiments ----------------------------- #

def run_experiments(specs: Iterable[MatrixSpec]) -> pd.DataFrame:
    records = []
    for spec in specs:
        matrix = make_low_rank_matrix(spec)
        (U_svd, S_svd, Vt_svd), svd_runtime = truncated_svd(matrix)
        for rank in RANKS_TO_TRY:
            approx = reconstruct(U_svd, S_svd, Vt_svd, rank)
            err = relative_frobenius_error(matrix, approx)
            records.append(
                {
                    "matrix": spec.name,
                    "method": "SVD",
                    "rank": rank,
                    "power_iterations": 0,
                    "oversampling": 0,
                    "relative_error": err,
                    "runtime_ms": svd_runtime * 1000,
                }
            )
        for rank in RANKS_TO_TRY:
            for q in POWER_ITERATIONS:
                (Ur, Sr, Vtr), runtime = randomized_svd(
                    matrix,
                    rank,
                    oversampling=OVERSAMPLING,
                    n_iter=q,
                    random_state=RANDOM_STATE,
                )
                approx = (Ur * Sr) @ Vtr
                err = relative_frobenius_error(matrix, approx)
                records.append(
                    {
                        "matrix": spec.name,
                        "method": "rSVD",
                        "rank": rank,
                        "power_iterations": q,
                        "oversampling": OVERSAMPLING,
                        "relative_error": err,
                        "runtime_ms": runtime * 1000,
                    }
                )
    df = pd.DataFrame(records)
    df.sort_values(["matrix", "method", "rank", "power_iterations"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["method", "rank", "power_iterations"], as_index=False)
        .agg(
            relative_error_mean=("relative_error", "mean"),
            relative_error_std=("relative_error", "std"),
            runtime_ms_mean=("runtime_ms", "mean"),
            runtime_ms_std=("runtime_ms", "std"),
        )
    )
    return grouped


def emit_latex_tables(avg: pd.DataFrame) -> None:
    latex_float = lambda x: f"{x:.4f}"
    svd_table = (
        avg[avg["method"] == "SVD"]
        [["rank", "relative_error_mean", "runtime_ms_mean"]]
        .rename(
            columns={
                "rank": "Rank",
                "relative_error_mean": "Relative Error",
                "runtime_ms_mean": "Runtime [ms]",
            }
        )
    )
    print(
        svd_table.to_latex(
            index=False,
            float_format=latex_float,
            caption="Deterministic SVD averages over synthetic matrices",
            label="tab:synthetic_svd",
        )
    )

    rsvd_table = (
        avg[avg["method"] == "rSVD"]
        [["rank", "power_iterations", "relative_error_mean", "runtime_ms_mean"]]
        .rename(
            columns={
                "rank": "Rank",
                "power_iterations": "q",
                "relative_error_mean": "Relative Error",
                "runtime_ms_mean": "Runtime [ms]",
            }
        )
    )
    print(
        rsvd_table.to_latex(
            index=False,
            float_format=latex_float,
            caption="Randomized SVD averages (mean over 50 matrices)",
            label="tab:synthetic_rsvd",
        )
    )

    best_rsvd = (
        rsvd_table.sort_values(["Rank", "Relative Error"])
        .groupby("Rank", as_index=False)
        .first()
    )
    comparison = best_rsvd.merge(
        svd_table,
        on="Rank",
        suffixes=("_rSVD", "_SVD"),
    )
    comparison["Error gain"] = (
        comparison["Relative Error_SVD"] - comparison["Relative Error_rSVD"]
    )
    comparison["Runtime ratio (SVD/rSVD)"] = (
        comparison["Runtime [ms]_SVD"] / comparison["Runtime [ms]_rSVD"]
    )
    print(
        comparison.to_latex(
            index=False,
            float_format=latex_float,
            caption="Best randomized SVD (by error) vs deterministic baseline",
            label="tab:synthetic_comparison",
        )
    )


def plot_mean_curves(avg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    error_ax, runtime_ax = axes
    svd_avg = avg[avg["method"] == "SVD"]
    error_ax.plot(svd_avg["rank"], svd_avg["relative_error_mean"], marker="o", label="SVD")
    runtime_ax.plot(svd_avg["rank"], svd_avg["runtime_ms_mean"], marker="o", label="SVD")
    for q, subset in avg[avg["method"] == "rSVD"].groupby("power_iterations"):
        label = f"rSVD (q={q})"
        error_ax.plot(subset["rank"], subset["relative_error_mean"], marker="o", label=label)
        runtime_ax.plot(subset["rank"], subset["runtime_ms_mean"], marker="o", label=label)
    error_ax.set_title("Mean Relative Error across 50 matrices")
    error_ax.set_xlabel("Rank")
    error_ax.set_ylabel("||A - A_k||_F / ||A||_F")
    error_ax.grid(True, linestyle=":", linewidth=0.7)
    error_ax.legend()
    runtime_ax.set_title("Mean Runtime across 50 matrices")
    runtime_ax.set_xlabel("Rank")
    runtime_ax.set_ylabel("Time (ms)")
    runtime_ax.grid(True, linestyle=":", linewidth=0.7)
    runtime_ax.legend()
    plt.tight_layout()
    path = FIGURE_DIR / "synthetic_mean_curves.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"Saved figure -> {path}")


# --------------------------------- main --------------------------------- #

def main() -> None:
    specs = generate_specs(NUM_MATRICES)
    print(f"Generated {len(specs)} matrix specs. First three:")
    for spec in specs[:3]:
        print(spec)

    metrics_df = run_experiments(specs)
    metrics_path = FIGURE_DIR / "synthetic_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote raw metrics -> {metrics_path}")

    avg_metrics = aggregate_metrics(metrics_df)
    avg_path = FIGURE_DIR / "synthetic_metrics_mean.csv"
    avg_metrics.to_csv(avg_path, index=False)
    print(f"Wrote aggregated metrics -> {avg_path}")

    emit_latex_tables(avg_metrics)
    plot_mean_curves(avg_metrics)


if __name__ == "__main__":
    main()
