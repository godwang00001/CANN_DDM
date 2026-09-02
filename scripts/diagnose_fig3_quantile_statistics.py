#!/usr/bin/env python3
"""Narrow diagnostics for Figure 3 quantile-probability RT statistics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QP_QUANTILES = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
DENSE_PERCENTILES = np.asarray([float(x) for x in np.linspace(0.05, 0.95, 19)], dtype=float)


@dataclass(frozen=True)
class RunData:
    run_root: Path
    config: dict[str, Any]
    metadata: dict[str, Any]
    model_names: list[str]
    coherence_values: np.ndarray
    choice: np.ndarray
    hit_boundary: np.ndarray
    rt_ms: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("results/figure3/fig3_psychometric_newdefaults"),
    )
    parser.add_argument(
        "--coherences",
        type=str,
        default="0.25,0.5",
        help="Comma-separated drift rates / coherences to inspect.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def parse_coherences(raw: str) -> np.ndarray:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("coherences must contain at least one numeric value")
    return np.asarray(values, dtype=float)


def load_run(run_root: Path) -> RunData:
    run_root = Path(run_root)
    config = json.loads((run_root / "config.json").read_text())
    with np.load(run_root / "dataset.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return RunData(
            run_root=run_root,
            config=config,
            metadata=metadata,
            model_names=[str(name) for name in data["model_names"].tolist()],
            coherence_values=np.asarray(data["coherence_values"], dtype=float),
            choice=np.asarray(data["choice"]),
            hit_boundary=np.asarray(data["hit_boundary"], dtype=bool),
            rt_ms=np.asarray(data["rt_ms"], dtype=float),
        )


def correct_choice_value(coherence: float) -> int:
    if float(coherence) > 0.0:
        return 1
    if float(coherence) < 0.0:
        return -1
    return 1


def coherence_index(run: RunData, coherence: float) -> int:
    matches = np.where(np.isclose(run.coherence_values, float(coherence)))[0]
    if matches.size == 0:
        raise ValueError(f"Coherence {coherence} not found in run {run.run_root}")
    return int(matches[0])


def outcome_rt(choice: np.ndarray, hit_boundary: np.ndarray, rt_ms: np.ndarray, coherence: float, outcome: str) -> np.ndarray:
    valid = np.asarray(hit_boundary, dtype=bool) & np.isfinite(np.asarray(rt_ms, dtype=float))
    choice_valid = np.asarray(choice[valid], dtype=int)
    rt_valid = np.asarray(rt_ms[valid], dtype=float)
    correct_choice = correct_choice_value(float(coherence))
    if outcome == "correct":
        mask = choice_valid == correct_choice
    else:
        mask = choice_valid == -correct_choice
    return np.asarray(rt_valid[mask], dtype=float)


def summary_rows(run: RunData, selected: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(run.model_names):
        for coherence in selected:
            idx = coherence_index(run, float(coherence))
            for outcome in ("correct", "error"):
                rt = outcome_rt(
                    run.choice[model_index, idx],
                    run.hit_boundary[model_index, idx],
                    run.rt_ms[model_index, idx],
                    float(coherence),
                    outcome,
                )
                rows.append(
                    {
                        "model": model_name,
                        "coherence": float(coherence),
                        "outcome": outcome,
                        "count": int(rt.size),
                        "mean_rt_ms": float(np.mean(rt)) if rt.size else float("nan"),
                        "median_rt_ms": float(np.median(rt)) if rt.size else float("nan"),
                        "min_rt_ms": float(np.min(rt)) if rt.size else float("nan"),
                        "max_rt_ms": float(np.max(rt)) if rt.size else float("nan"),
                    }
                )
    return pd.DataFrame(rows).sort_values(["coherence", "outcome", "model"]).reset_index(drop=True)


def quantile_rows(run: RunData, selected: np.ndarray, quantiles: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(run.model_names):
        for coherence in selected:
            idx = coherence_index(run, float(coherence))
            for outcome in ("correct", "error"):
                rt = outcome_rt(
                    run.choice[model_index, idx],
                    run.hit_boundary[model_index, idx],
                    run.rt_ms[model_index, idx],
                    float(coherence),
                    outcome,
                )
                if rt.size == 0:
                    continue
                for quantile in np.asarray(quantiles, dtype=float):
                    rows.append(
                        {
                            "model": model_name,
                            "coherence": float(coherence),
                            "outcome": outcome,
                            "quantile": float(quantile),
                            "rt_quantile_ms": float(np.quantile(rt, float(quantile))),
                            "count": int(rt.size),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["coherence", "outcome", "model", "quantile"]).reset_index(drop=True)


def rank_curve_table(dense_quantiles: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        dense_quantiles.pivot_table(
            index=["coherence", "outcome", "quantile"],
            columns="model",
            values="rt_quantile_ms",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot["circuit_minus_ddm"] = pivot["circuit"] - pivot["ddm"]
    return pivot.sort_values(["coherence", "outcome", "quantile"]).reset_index(drop=True)


def quantile_impl_check(notebook_like: pd.DataFrame, rank_curve: pd.DataFrame) -> pd.DataFrame:
    merged = notebook_like.merge(
        rank_curve,
        on=["coherence", "outcome", "quantile"],
        how="left",
        suffixes=("", "_pivot"),
    )
    merged["expected_from_rank_curve"] = merged.apply(
        lambda row: row["ddm"] if row["model"] == "ddm" else row["circuit"],
        axis=1,
    )
    merged["abs_diff_ms"] = np.abs(merged["rt_quantile_ms"] - merged["expected_from_rank_curve"])
    return merged[
        [
            "model",
            "coherence",
            "outcome",
            "quantile",
            "rt_quantile_ms",
            "expected_from_rank_curve",
            "abs_diff_ms",
            "count",
        ]
    ].sort_values(["coherence", "outcome", "model", "quantile"]).reset_index(drop=True)


def save_percentile_overlay(dense_quantiles: pd.DataFrame, notebook_like: pd.DataFrame, output_path: Path) -> None:
    coherences = sorted(dense_quantiles["coherence"].unique().tolist())
    outcomes = ["correct", "error"]
    fig, axes = plt.subplots(len(outcomes), len(coherences), figsize=(5.0 * len(coherences), 7.2), sharex=False, sharey=False)
    if len(outcomes) == 1 and len(coherences) == 1:
        axes = np.asarray([[axes]])
    elif len(outcomes) == 1:
        axes = np.asarray([axes])
    elif len(coherences) == 1:
        axes = np.asarray([[ax] for ax in axes])

    colors = {"ddm": "#1d4ed8", "circuit": "#b91c1c"}
    markers = {0.1: "o", 0.3: "s", 0.5: "^", 0.7: "D", 0.9: "P"}

    for row_index, outcome in enumerate(outcomes):
        for col_index, coherence in enumerate(coherences):
            ax = axes[row_index, col_index]
            dense_slice = dense_quantiles[
                (np.isclose(dense_quantiles["coherence"], coherence))
                & (dense_quantiles["outcome"] == outcome)
            ].copy()
            notebook_slice = notebook_like[
                (np.isclose(notebook_like["coherence"], coherence))
                & (notebook_like["outcome"] == outcome)
            ].copy()
            for model_name in ("ddm", "circuit"):
                model_dense = dense_slice[dense_slice["model"] == model_name].sort_values("quantile")
                model_qp = notebook_slice[notebook_slice["model"] == model_name].sort_values("quantile")
                ax.plot(model_dense["quantile"], model_dense["rt_quantile_ms"], color=colors[model_name], linewidth=2.0, label=f"{model_name} dense")
                for _, qp_row in model_qp.iterrows():
                    ax.scatter(
                        [qp_row["quantile"]],
                        [qp_row["rt_quantile_ms"]],
                        color=colors[model_name],
                        marker=markers[float(qp_row["quantile"])],
                        s=60,
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=3,
                    )
            ax.set_title(f"coh={coherence:g}, {outcome}")
            ax.set_xlabel("Quantile")
            ax.set_ylabel("RT (ms)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_rank_scatter(rank_curve: pd.DataFrame, output_path: Path) -> None:
    coherences = sorted(rank_curve["coherence"].unique().tolist())
    outcomes = ["correct", "error"]
    fig, axes = plt.subplots(len(outcomes), len(coherences), figsize=(5.0 * len(coherences), 7.2), sharex=False, sharey=False)
    if len(outcomes) == 1 and len(coherences) == 1:
        axes = np.asarray([[axes]])
    elif len(outcomes) == 1:
        axes = np.asarray([axes])
    elif len(coherences) == 1:
        axes = np.asarray([[ax] for ax in axes])

    for row_index, outcome in enumerate(outcomes):
        for col_index, coherence in enumerate(coherences):
            ax = axes[row_index, col_index]
            subset = rank_curve[(np.isclose(rank_curve["coherence"], coherence)) & (rank_curve["outcome"] == outcome)].copy()
            ax.scatter(subset["ddm"], subset["circuit"], c=subset["quantile"], cmap="viridis", s=40)
            lo = float(np.nanmin([subset["ddm"].min(), subset["circuit"].min()]))
            hi = float(np.nanmax([subset["ddm"].max(), subset["circuit"].max()]))
            ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
            ax.set_title(f"coh={coherence:g}, {outcome}")
            ax.set_xlabel("DDM percentile RT (ms)")
            ax.set_ylabel("Circuit percentile RT (ms)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    output_path: Path,
    run: RunData,
    selected: np.ndarray,
    summary_df: pd.DataFrame,
    rank_curve: pd.DataFrame,
    impl_check: pd.DataFrame,
) -> None:
    lines = [
        "# Fig3 Quantile Statistics Narrow Diagnostic",
        "",
        f"Run: `{run.run_root}`",
        f"Selected coherences: `{', '.join(f'{value:g}' for value in selected)}`",
        "",
        "## Main Checks",
        "- This diagnostic does not pair DDM and circuit trial indices, because the saved trial seeds differ between models.",
        "- It checks whether the quantile-probability points are direct quantiles of the saved RT arrays after the same hit/outcome filtering.",
        "- It also compares dense percentile curves between DDM and circuit to show whether the mismatch is already present in the raw RT distributions.",
        "",
        "## Outcome RT Summary",
        summary_df.to_markdown(index=False),
        "",
        "## Quantile Implementation Check",
        f"- Maximum absolute difference between notebook-style quantile points and the same values recovered from the dense rank curve table: `{float(np.nanmax(impl_check['abs_diff_ms'])):.6f} ms`",
        "",
        "## Rank-Curve Delta Summary",
        rank_curve.groupby(['coherence', 'outcome'], dropna=False)['circuit_minus_ddm'].agg(['min', 'median', 'max']).reset_index().to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    run = load_run(args.run_root)
    selected = parse_coherences(args.coherences)
    output_dir = args.output_dir or (run.run_root / "quantile_narrow_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = summary_rows(run, selected)
    notebook_like = quantile_rows(run, selected, QP_QUANTILES)
    dense_quantiles = quantile_rows(run, selected, DENSE_PERCENTILES)
    rank_curve = rank_curve_table(dense_quantiles)
    impl_check = quantile_impl_check(notebook_like, rank_curve_table(quantile_rows(run, selected, QP_QUANTILES)))

    summary_df.to_csv(output_dir / "outcome_rt_summary.csv", index=False)
    notebook_like.to_csv(output_dir / "notebook_quantile_points.csv", index=False)
    dense_quantiles.to_csv(output_dir / "dense_percentile_curves.csv", index=False)
    rank_curve.to_csv(output_dir / "rank_curve_comparison.csv", index=False)
    impl_check.to_csv(output_dir / "quantile_implementation_check.csv", index=False)

    save_percentile_overlay(dense_quantiles, notebook_like, output_dir / "percentile_overlay.png")
    save_rank_scatter(rank_curve, output_dir / "rank_scatter.png")
    write_report(
        output_path=output_dir / "report.md",
        run=run,
        selected=selected,
        summary_df=summary_df,
        rank_curve=rank_curve,
        impl_check=impl_check,
    )

    print(f"Saved narrow diagnostic to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
