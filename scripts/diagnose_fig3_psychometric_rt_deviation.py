#!/usr/bin/env python3
"""Diagnose RT deviations between Figure 3 psychometric runs."""

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
QP_DRIFT_RATES = np.asarray([0.125, 0.25, 0.5, 1.0], dtype=float)


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
    final_x: np.ndarray
    time_ms: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-run",
        type=Path,
        default=Path("results/figure3/fig3_psychometric_test_split"),
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        default=Path("results/figure3/fig3_psychometric_c0p5_n1000_dur4000"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    return parser.parse_args()


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
            final_x=np.asarray(data["final_x"], dtype=float),
            time_ms=np.asarray(data["time_ms"], dtype=float),
        )


def summarize_model_sweep(run: RunData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(run.model_names):
        for coherence_index, coherence in enumerate(run.coherence_values):
            choice_row = run.choice[model_index, coherence_index]
            hit_row = run.hit_boundary[model_index, coherence_index]
            rt_row = run.rt_ms[model_index, coherence_index]
            valid_rt = rt_row[hit_row & np.isfinite(rt_row)]
            num_hit = int(np.sum(hit_row))
            p_right = float(np.mean(choice_row[hit_row] == 1)) if num_hit > 0 else float("nan")
            rows.append(
                {
                    "model": model_name,
                    "coherence": float(coherence),
                    "mean_rt_ms": float(np.mean(valid_rt)) if valid_rt.size else float("nan"),
                    "median_rt_ms": float(np.median(valid_rt)) if valid_rt.size else float("nan"),
                    "p_right": p_right,
                    "num_hit": num_hit,
                    "miss_fraction": 1.0 - float(np.mean(hit_row)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "coherence"]).reset_index(drop=True)


def outcome_rt_rows(run: RunData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(run.model_names):
        for coherence_index, coherence in enumerate(run.coherence_values):
            hit_mask = run.hit_boundary[model_index, coherence_index] & np.isfinite(run.rt_ms[model_index, coherence_index])
            choice = run.choice[model_index, coherence_index][hit_mask]
            rt = run.rt_ms[model_index, coherence_index][hit_mask]
            for outcome_name, outcome_mask in (("correct", choice == 1), ("error", choice != 1)):
                outcome_rt = rt[outcome_mask]
                rows.append(
                    {
                        "model": model_name,
                        "coherence": float(coherence),
                        "outcome": outcome_name,
                        "count": int(outcome_rt.size),
                        "mean_rt_ms": float(np.mean(outcome_rt)) if outcome_rt.size else float("nan"),
                        "median_rt_ms": float(np.median(outcome_rt)) if outcome_rt.size else float("nan"),
                    }
                )
    return pd.DataFrame(rows).sort_values(["model", "coherence", "outcome"]).reset_index(drop=True)


def quantile_probability_rows(run: RunData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(run.model_names):
        choice = run.choice[model_index]
        hit_boundary = run.hit_boundary[model_index]
        rt_ms = run.rt_ms[model_index]
        for coherence_index, coherence in enumerate(run.coherence_values):
            if not np.any(np.isclose(float(coherence), QP_DRIFT_RATES)):
                continue
            valid_mask = hit_boundary[coherence_index] & np.isfinite(rt_ms[coherence_index])
            if not np.any(valid_mask):
                continue
            choice_valid = choice[coherence_index][valid_mask]
            rt_valid = rt_ms[coherence_index][valid_mask]
            total_count = int(rt_valid.size)
            for outcome_name, outcome_mask in (("correct", choice_valid == 1), ("error", choice_valid != 1)):
                outcome_rt = rt_valid[outcome_mask]
                outcome_count = int(outcome_rt.size)
                if outcome_count < 1:
                    continue
                response_proportion = float(outcome_count) / float(total_count)
                for quantile in QP_QUANTILES:
                    rows.append(
                        {
                            "model": model_name,
                            "coherence": float(coherence),
                            "outcome": outcome_name,
                            "quantile": float(quantile),
                            "response_proportion": response_proportion,
                            "rt_quantile_ms": float(np.quantile(outcome_rt, float(quantile))),
                            "count": outcome_count,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["model", "coherence", "outcome", "quantile"]).reset_index(drop=True)


def extract_condition_seeds(metadata: dict[str, Any], model_name: str) -> list[int | None]:
    model_metadata = metadata.get(f"{model_name}_metadata", {})
    if "condition_seeds" in model_metadata:
        return [int(seed) for seed in model_metadata["condition_seeds"]]
    if model_name == "circuit":
        seeds: list[int | None] = []
        for condition in model_metadata.get("condition_metadata", []):
            if "seed" in condition:
                seeds.append(int(condition["seed"]))
                continue
            batch_metadata = condition.get("batch_metadata", [])
            if batch_metadata:
                seeds.append(int(batch_metadata[0]["seed"]))
            else:
                seeds.append(None)
        return seeds
    return []


def extract_calibration(metadata: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    circuit_metadata = metadata.get("circuit_metadata", {})
    if "calibration" in circuit_metadata:
        return dict(circuit_metadata["calibration"])
    circuit_config = metadata.get("circuit_config", {})
    if "calibration" in circuit_config:
        return dict(circuit_config["calibration"])
    if "calibration" in config:
        return dict(config["calibration"])
    return {}


def compare_overall_mean_rt(run_label: str, summary_df: pd.DataFrame) -> pd.DataFrame:
    pivot = summary_df.pivot(index="coherence", columns="model", values="mean_rt_ms").reset_index()
    pivot["run"] = run_label
    pivot["circuit_minus_ddm"] = pivot["circuit"] - pivot["ddm"]
    return pivot[["run", "coherence", "ddm", "circuit", "circuit_minus_ddm"]]


def compare_outcome_rt(run_label: str, outcome_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        outcome_df.pivot_table(index=["coherence", "outcome"], columns="model", values="mean_rt_ms")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot["run"] = run_label
    pivot["circuit_minus_ddm"] = pivot["circuit"] - pivot["ddm"]
    return pivot[["run", "coherence", "outcome", "ddm", "circuit", "circuit_minus_ddm"]]


def compare_quantiles(run_label: str, quantile_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        quantile_df.pivot_table(
            index=["coherence", "outcome", "quantile", "response_proportion"],
            columns="model",
            values="rt_quantile_ms",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot["run"] = run_label
    pivot["circuit_minus_ddm"] = pivot["circuit"] - pivot["ddm"]
    return pivot[["run", "coherence", "outcome", "quantile", "response_proportion", "ddm", "circuit", "circuit_minus_ddm"]]


def compare_runs(current: pd.DataFrame, baseline: pd.DataFrame, join_cols: list[str], value_cols: list[str], current_label: str, baseline_label: str) -> pd.DataFrame:
    merged = current.merge(baseline, on=join_cols, suffixes=(f"_{current_label}", f"_{baseline_label}"))
    for value_col in value_cols:
        merged[f"{value_col}_delta"] = merged[f"{value_col}_{current_label}"] - merged[f"{value_col}_{baseline_label}"]
    return merged


def bootstrap_delta(current_values: np.ndarray, baseline_values: np.ndarray, statistic: str, quantile: float | None, rng: np.random.Generator, samples: int) -> tuple[float, float, float]:
    current_values = np.asarray(current_values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    if current_values.size == 0 or baseline_values.size == 0:
        return float("nan"), float("nan"), float("nan")
    deltas = np.empty(samples, dtype=float)
    for index in range(samples):
        sample_current = rng.choice(current_values, size=current_values.size, replace=True)
        sample_baseline = rng.choice(baseline_values, size=baseline_values.size, replace=True)
        if statistic == "mean":
            deltas[index] = float(np.mean(sample_current) - np.mean(sample_baseline))
        else:
            assert quantile is not None
            deltas[index] = float(np.quantile(sample_current, quantile) - np.quantile(sample_baseline, quantile))
    return float(np.quantile(deltas, 0.025)), float(np.median(deltas)), float(np.quantile(deltas, 0.975))


def bootstrap_tables(run: RunData, samples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mean_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    ddm_index = run.model_names.index("ddm")
    circuit_index = run.model_names.index("circuit")
    for coherence_index, coherence in enumerate(run.coherence_values):
        ddm_valid = run.hit_boundary[ddm_index, coherence_index] & np.isfinite(run.rt_ms[ddm_index, coherence_index])
        circuit_valid = run.hit_boundary[circuit_index, coherence_index] & np.isfinite(run.rt_ms[circuit_index, coherence_index])
        ddm_rt = run.rt_ms[ddm_index, coherence_index][ddm_valid]
        circuit_rt = run.rt_ms[circuit_index, coherence_index][circuit_valid]
        mean_ci_low, mean_median, mean_ci_high = bootstrap_delta(circuit_rt, ddm_rt, "mean", None, rng, samples)
        observed_mean_delta = float(np.mean(circuit_rt) - np.mean(ddm_rt)) if ddm_rt.size and circuit_rt.size else float("nan")
        mean_rows.append(
            {
                "coherence": float(coherence),
                "outcome": "all_hits",
                "observed_delta_ms": observed_mean_delta,
                "bootstrap_ci_low_ms": mean_ci_low,
                "bootstrap_median_ms": mean_median,
                "bootstrap_ci_high_ms": mean_ci_high,
                "covers_zero": bool(mean_ci_low <= 0.0 <= mean_ci_high) if np.isfinite(mean_ci_low) else False,
                "ddm_count": int(ddm_rt.size),
                "circuit_count": int(circuit_rt.size),
            }
        )
        ddm_choice = run.choice[ddm_index, coherence_index][ddm_valid]
        circuit_choice = run.choice[circuit_index, coherence_index][circuit_valid]
        for outcome_name, ddm_mask, circuit_mask in (
            ("correct", ddm_choice == 1, circuit_choice == 1),
            ("error", ddm_choice != 1, circuit_choice != 1),
        ):
            ddm_outcome = ddm_rt[ddm_mask]
            circuit_outcome = circuit_rt[circuit_mask]
            mean_ci_low, mean_median, mean_ci_high = bootstrap_delta(circuit_outcome, ddm_outcome, "mean", None, rng, samples)
            observed_delta = float(np.mean(circuit_outcome) - np.mean(ddm_outcome)) if ddm_outcome.size and circuit_outcome.size else float("nan")
            mean_rows.append(
                {
                    "coherence": float(coherence),
                    "outcome": outcome_name,
                    "observed_delta_ms": observed_delta,
                    "bootstrap_ci_low_ms": mean_ci_low,
                    "bootstrap_median_ms": mean_median,
                    "bootstrap_ci_high_ms": mean_ci_high,
                    "covers_zero": bool(mean_ci_low <= 0.0 <= mean_ci_high) if np.isfinite(mean_ci_low) else False,
                    "ddm_count": int(ddm_outcome.size),
                    "circuit_count": int(circuit_outcome.size),
                }
            )
            if not np.any(np.isclose(float(coherence), QP_DRIFT_RATES)):
                continue
            for quantile in QP_QUANTILES:
                q_ci_low, q_median, q_ci_high = bootstrap_delta(circuit_outcome, ddm_outcome, "quantile", float(quantile), rng, samples)
                observed_quantile_delta = (
                    float(np.quantile(circuit_outcome, float(quantile)) - np.quantile(ddm_outcome, float(quantile)))
                    if ddm_outcome.size and circuit_outcome.size
                    else float("nan")
                )
                quantile_rows.append(
                    {
                        "coherence": float(coherence),
                        "outcome": outcome_name,
                        "quantile": float(quantile),
                        "observed_delta_ms": observed_quantile_delta,
                        "bootstrap_ci_low_ms": q_ci_low,
                        "bootstrap_median_ms": q_median,
                        "bootstrap_ci_high_ms": q_ci_high,
                        "covers_zero": bool(q_ci_low <= 0.0 <= q_ci_high) if np.isfinite(q_ci_low) else False,
                        "ddm_count": int(ddm_outcome.size),
                        "circuit_count": int(circuit_outcome.size),
                    }
                )
    return pd.DataFrame(mean_rows), pd.DataFrame(quantile_rows)


def build_metadata_comparison(current: RunData, baseline: RunData) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_rows = [
        {"run": current.run_root.name, **extract_calibration(current.metadata, current.config)},
        {"run": baseline.run_root.name, **extract_calibration(baseline.metadata, baseline.config)},
    ]
    calibration_df = pd.DataFrame(calibration_rows)

    rows: list[dict[str, Any]] = []
    current_ddm_seeds = extract_condition_seeds(current.metadata, "ddm")
    baseline_ddm_seeds = extract_condition_seeds(baseline.metadata, "ddm")
    current_circuit_seeds = extract_condition_seeds(current.metadata, "circuit")
    baseline_circuit_seeds = extract_condition_seeds(baseline.metadata, "circuit")
    for index, coherence in enumerate(current.coherence_values):
        rows.append(
            {
                "coherence": float(coherence),
                "current_ddm_seed": current_ddm_seeds[index] if index < len(current_ddm_seeds) else None,
                "baseline_ddm_seed": baseline_ddm_seeds[index] if index < len(baseline_ddm_seeds) else None,
                "current_circuit_seed": current_circuit_seeds[index] if index < len(current_circuit_seeds) else None,
                "baseline_circuit_seed": baseline_circuit_seeds[index] if index < len(baseline_circuit_seeds) else None,
            }
        )
    return calibration_df, pd.DataFrame(rows)


def classify_findings(mean_bootstrap_df: pd.DataFrame, calibration_df: pd.DataFrame, seed_df: pd.DataFrame) -> dict[str, Any]:
    all_hits = mean_bootstrap_df[mean_bootstrap_df["outcome"] == "all_hits"].copy()
    systematic_count = int(np.sum(~all_hits["covers_zero"]))
    median_abs_delta = float(np.nanmedian(np.abs(all_hits["observed_delta_ms"])))
    calibration_changed = False
    if len(calibration_df) == 2 and {"kappa", "c_be_theta_max"}.issubset(calibration_df.columns):
        calibration_changed = bool(
            not np.isclose(float(calibration_df.iloc[0]["kappa"]), float(calibration_df.iloc[1]["kappa"]))
            or not np.isclose(float(calibration_df.iloc[0]["c_be_theta_max"]), float(calibration_df.iloc[1]["c_be_theta_max"]))
        )
    seeds_changed = bool(
        (seed_df["current_ddm_seed"] != seed_df["baseline_ddm_seed"]).any()
        or (seed_df["current_circuit_seed"] != seed_df["baseline_circuit_seed"]).any()
    )
    if systematic_count >= 5 and calibration_changed:
        label = "systematic-calibration"
    elif systematic_count >= 5 and seeds_changed:
        label = "systematic-seed-path"
    elif systematic_count >= 3:
        label = "mixed-but-systematic"
    else:
        label = "sampling-dominated"
    return {
        "label": label,
        "systematic_coherence_count": systematic_count,
        "median_abs_mean_rt_delta_ms": median_abs_delta,
        "calibration_changed": calibration_changed,
        "seeds_changed": seeds_changed,
    }


def save_plot(df: pd.DataFrame, output_path: Path, *, title: str, y_col: str, group_cols: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
        for group_key, group_df in grouped:
            label = ", ".join(str(item) for item in (group_key if isinstance(group_key, tuple) else (group_key,)))
            group_df = group_df.sort_values("coherence")
            ax.plot(group_df["coherence"], group_df[y_col], marker="o", linewidth=1.8, label=label)
    else:
        group_df = df.sort_values("coherence")
        ax.plot(group_df["coherence"], group_df[y_col], marker="o", linewidth=1.8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Coherence")
    ax.set_ylabel("Delta RT (ms)")
    ax.set_title(title)
    if group_cols:
        ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    output_path: Path,
    current_run: RunData,
    baseline_run: RunData,
    classification: dict[str, Any],
    overall_current: pd.DataFrame,
    overall_baseline: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> None:
    lines = [
        "# Fig3 RT Deviation Diagnostic Report",
        "",
        f"Current run: `{current_run.run_root}`",
        f"Baseline run: `{baseline_run.run_root}`",
        "",
        "## Classification",
        f"- Label: `{classification['label']}`",
        f"- Significant coherence count (all-hit bootstrap): `{classification['systematic_coherence_count']}`",
        f"- Median absolute mean-RT delta: `{classification['median_abs_mean_rt_delta_ms']:.2f} ms`",
        f"- Calibration changed: `{classification['calibration_changed']}`",
        f"- Seeds changed: `{classification['seeds_changed']}`",
        "",
        "## Calibration Comparison",
        calibration_df.to_markdown(index=False),
        "",
        "## Overall Mean RT Delta",
        "Current run:",
        overall_current.to_markdown(index=False),
        "",
        "Baseline run:",
        overall_baseline.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    current_run = load_run(args.current_run)
    baseline_run = load_run(args.baseline_run)

    if args.output_dir is None:
        output_dir = current_run.run_root / f"diagnostics_vs_{baseline_run.run_root.name}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_summary = summarize_model_sweep(current_run)
    baseline_summary = summarize_model_sweep(baseline_run)
    current_outcomes = outcome_rt_rows(current_run)
    baseline_outcomes = outcome_rt_rows(baseline_run)
    current_quantiles = quantile_probability_rows(current_run)
    baseline_quantiles = quantile_probability_rows(baseline_run)

    overall_current = compare_overall_mean_rt(current_run.run_root.name, current_summary)
    overall_baseline = compare_overall_mean_rt(baseline_run.run_root.name, baseline_summary)
    outcome_current = compare_outcome_rt(current_run.run_root.name, current_outcomes)
    outcome_baseline = compare_outcome_rt(baseline_run.run_root.name, baseline_outcomes)
    quantile_current = compare_quantiles(current_run.run_root.name, current_quantiles)
    quantile_baseline = compare_quantiles(baseline_run.run_root.name, baseline_quantiles)

    run_delta_overall = compare_runs(
        overall_current,
        overall_baseline,
        ["coherence"],
        ["ddm", "circuit", "circuit_minus_ddm"],
        current_run.run_root.name,
        baseline_run.run_root.name,
    )
    run_delta_outcome = compare_runs(
        outcome_current,
        outcome_baseline,
        ["coherence", "outcome"],
        ["ddm", "circuit", "circuit_minus_ddm"],
        current_run.run_root.name,
        baseline_run.run_root.name,
    )
    run_delta_quantile = compare_runs(
        quantile_current,
        quantile_baseline,
        ["coherence", "outcome", "quantile", "response_proportion"],
        ["ddm", "circuit", "circuit_minus_ddm"],
        current_run.run_root.name,
        baseline_run.run_root.name,
    )

    bootstrap_mean_df, bootstrap_quantile_df = bootstrap_tables(
        current_run,
        samples=int(args.bootstrap_samples),
        seed=int(args.bootstrap_seed),
    )
    calibration_df, seed_df = build_metadata_comparison(current_run, baseline_run)
    classification = classify_findings(bootstrap_mean_df, calibration_df, seed_df)

    current_summary.to_csv(output_dir / "current_summary.csv", index=False)
    baseline_summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    current_outcomes.to_csv(output_dir / "current_outcome_rt.csv", index=False)
    baseline_outcomes.to_csv(output_dir / "baseline_outcome_rt.csv", index=False)
    current_quantiles.to_csv(output_dir / "current_quantile_probability.csv", index=False)
    baseline_quantiles.to_csv(output_dir / "baseline_quantile_probability.csv", index=False)
    overall_current.to_csv(output_dir / "current_overall_mean_rt_delta.csv", index=False)
    overall_baseline.to_csv(output_dir / "baseline_overall_mean_rt_delta.csv", index=False)
    outcome_current.to_csv(output_dir / "current_outcome_mean_rt_delta.csv", index=False)
    outcome_baseline.to_csv(output_dir / "baseline_outcome_mean_rt_delta.csv", index=False)
    quantile_current.to_csv(output_dir / "current_quantile_rt_delta.csv", index=False)
    quantile_baseline.to_csv(output_dir / "baseline_quantile_rt_delta.csv", index=False)
    run_delta_overall.to_csv(output_dir / "run_to_run_overall_delta.csv", index=False)
    run_delta_outcome.to_csv(output_dir / "run_to_run_outcome_delta.csv", index=False)
    run_delta_quantile.to_csv(output_dir / "run_to_run_quantile_delta.csv", index=False)
    bootstrap_mean_df.to_csv(output_dir / "bootstrap_mean_rt_delta.csv", index=False)
    bootstrap_quantile_df.to_csv(output_dir / "bootstrap_quantile_rt_delta.csv", index=False)
    calibration_df.to_csv(output_dir / "calibration_comparison.csv", index=False)
    seed_df.to_csv(output_dir / "seed_comparison.csv", index=False)
    (output_dir / "classification.json").write_text(json.dumps(classification, indent=2))

    save_plot(
        overall_current,
        output_dir / "current_overall_mean_rt_delta.png",
        title=f"{current_run.run_root.name}: circuit - DDM mean RT",
        y_col="circuit_minus_ddm",
        group_cols=["run"],
    )
    save_plot(
        outcome_current,
        output_dir / "current_outcome_mean_rt_delta.png",
        title=f"{current_run.run_root.name}: circuit - DDM outcome RT",
        y_col="circuit_minus_ddm",
        group_cols=["outcome"],
    )
    save_plot(
        run_delta_overall,
        output_dir / "run_to_run_overall_delta.png",
        title=f"{current_run.run_root.name} - {baseline_run.run_root.name}: overall delta",
        y_col="circuit_minus_ddm_delta",
        group_cols=[],
    )
    write_report(
        output_path=output_dir / "report.md",
        current_run=current_run,
        baseline_run=baseline_run,
        classification=classification,
        overall_current=overall_current,
        overall_baseline=overall_baseline,
        calibration_df=calibration_df,
    )

    print(f"Saved diagnostics to {output_dir}")
    print(f"Classification: {classification['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
