#!/usr/bin/env python3
"""Generate psychometric-condition simulation data for continuous DDM or circuit models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rate_model_core.accumulator_simulation import (
    ContinuousDDMConfig,
    load_simulation_result_npz,
    prepare_circuit_target_diffusion_calibration,
    save_simulation_result_npz,
    simulate_circuit_trials,
    simulate_ddm_trials,
)


def parse_coherence_values(raw: str) -> np.ndarray:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("coherence-values must contain at least one numeric value")
    return np.asarray(values, dtype=float)


def binomial_half_width(p_right: float, num_trials: int) -> float:
    if int(num_trials) <= 0:
        return float("nan")
    return 1.96 * np.sqrt(float(p_right) * (1.0 - float(p_right)) / float(num_trials))


def coherence_slug(coherence: float) -> str:
    sign = "p" if float(coherence) >= 0.0 else "m"
    magnitude = f"{abs(float(coherence)):.6f}".rstrip("0").rstrip(".")
    magnitude = magnitude if magnitude else "0"
    return f"{sign}{magnitude.replace('.', 'p')}"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("ddm", "circuit"), required=True)
    parser.add_argument("--coherence-values", type=str, required=True)
    parser.add_argument("--drift-gain", type=float, required=True)
    parser.add_argument("--noise-scale", type=float, required=True)
    parser.add_argument("--dt-ddm", type=float, required=True)
    parser.add_argument("--dt-model", type=float, default=1.0)
    parser.add_argument("--t-start", type=int, required=True)
    parser.add_argument("--dur", type=int, required=True)
    parser.add_argument("--max-time", type=int)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--boundary", type=float, default=1.0)
    parser.add_argument("--mar", type=float, default=0.0)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--decision-paradigm", choices=("free_response", "interrogation"), default="free_response")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-traj", action="store_true")
    return parser


def simulate_one_condition(
    *,
    model: str,
    coherence: float,
    drift_gain: float,
    noise_scale: float,
    dt_ddm: float,
    dt_model: float,
    t_start: int,
    dur: int,
    max_time: int | None,
    x0: float,
    boundary: float,
    mar: float,
    num_trials: int,
    seed: int,
    decision_paradigm: str,
    chunk_ms: int,
    calibration=None,
    save_traj: bool,
):
    drift_rate = float(drift_gain) * float(coherence)
    if model == "ddm":
        return simulate_ddm_trials(
            decision_mode="continuous",
            config=ContinuousDDMConfig(
                drift_rate=drift_rate,
                noise_scale=float(noise_scale),
            ),
            dt_DDM=float(dt_ddm),
            dur=int(dur),
            max_time=None if max_time is None else int(max_time),
            t_start=int(t_start),
            x0=float(x0),
            boundary=float(boundary),
            num_trials=int(num_trials),
            decision_paradigm=str(decision_paradigm),
            seed=int(seed),
            return_traj=bool(save_traj),
        )
    return simulate_circuit_trials(
        coherence=float(coherence),
        drift_gain=float(drift_gain),
        noise_scale=float(noise_scale),
        dt_ddm=float(dt_ddm),
        dt_model=float(dt_model),
        t_start=int(t_start),
        dur=int(dur),
        max_time=None if max_time is None else int(max_time),
        num_trials=int(num_trials),
        seed=int(seed),
        decision_paradigm=str(decision_paradigm),
        save_traj=bool(save_traj),
        chunk_ms=int(chunk_ms),
        calibration=calibration,
        x0=float(x0),
        boundary=float(boundary),
        mar=float(mar),
    )


def summarize_result(*, model: str, coherence: float, drift_gain: float, num_trials: int, seed: int, result_file: str, result) -> dict[str, object]:
    hit_mask = np.asarray(result.hit_boundary, dtype=bool)
    num_hit = int(np.sum(hit_mask))
    p_right = float(np.mean(result.choice[hit_mask] == 1)) if num_hit > 0 else float("nan")
    hit_fraction = float(np.mean(result.hit_boundary))
    mean_rt_ms = float(np.nanmean(result.rt_ms)) if np.any(result.hit_boundary) else float("nan")
    return {
        "model": str(model),
        "coherence": float(coherence),
        "drift_rate": float(drift_gain) * float(coherence),
        "p_right": p_right,
        "num_hit": int(num_hit),
        "miss_fraction": 1.0 - hit_fraction,
        "hit_fraction": hit_fraction,
        "mean_rt_ms": mean_rt_ms,
        "ci_half_width": binomial_half_width(p_right, int(num_hit)),
        "num_trials": int(num_trials),
        "seed": int(seed),
        "result_file": str(result_file),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    coherence_values = parse_coherence_values(args.coherence_values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "conditions"
    results_dir.mkdir(parents=True, exist_ok=True)

    seed_sequence = np.random.SeedSequence(int(args.seed))
    child_sequences = seed_sequence.spawn(len(coherence_values))

    summary_rows: list[dict[str, object]] = []
    calibration = None

    for coherence, child in zip(coherence_values, child_sequences):
        trial_seed = int(child.generate_state(1)[0])
        result_path = results_dir / f"{args.model}_coh_{coherence_slug(float(coherence))}.npz"
        if args.resume and result_path.exists():
            result = load_simulation_result_npz(result_path)
            if args.model == "circuit" and calibration is None and "kappa" in result.metadata:
                calibration = {
                    "kappa": float(result.metadata["kappa"]),
                    "certificate_passed": bool(result.metadata.get("certificate_passed", True)),
                    "c_be_theta_max": float(result.metadata.get("c_be_theta_max", float("nan"))),
                }
        else:
            if args.model == "circuit" and calibration is None:
                calibration = prepare_circuit_target_diffusion_calibration(
                    dt_ddm=float(args.dt_ddm),
                    t_start=int(args.t_start),
                    dur=int(args.dur),
                    max_time=None if args.max_time is None else int(args.max_time),
                    seed=int(args.seed),
                )
            result = simulate_one_condition(
                model=str(args.model),
                coherence=float(coherence),
                drift_gain=float(args.drift_gain),
                noise_scale=float(args.noise_scale),
                dt_ddm=float(args.dt_ddm),
                dt_model=float(args.dt_model),
                t_start=int(args.t_start),
                dur=int(args.dur),
                max_time=None if args.max_time is None else int(args.max_time),
                x0=float(args.x0),
                boundary=float(args.boundary),
                mar=float(args.mar),
                num_trials=int(args.num_trials),
                seed=trial_seed,
                decision_paradigm=str(args.decision_paradigm),
                chunk_ms=int(args.chunk_ms),
                calibration=calibration,
                save_traj=bool(args.save_traj),
            )
            save_simulation_result_npz(result_path, result)

        summary_rows.append(
            summarize_result(
                model=str(args.model),
                coherence=float(coherence),
                drift_gain=float(args.drift_gain),
                num_trials=int(args.num_trials),
                seed=trial_seed,
                result_file=str(result_path.relative_to(output_dir)),
                result=result,
            )
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    config_path = output_dir / "config.json"
    config_payload = {
        "model": str(args.model),
        "decision_mode": "continuous",
        "decision_paradigm": str(args.decision_paradigm),
        "coherence_values": coherence_values.tolist(),
        "drift_gain": float(args.drift_gain),
        "noise_scale": float(args.noise_scale),
        "dt_ddm": float(args.dt_ddm),
        "dt_model": float(args.dt_model),
        "t_start": int(args.t_start),
        "dur": int(args.dur),
        "max_time": int(args.max_time) if args.max_time is not None else int(args.dur),
        "x0": float(args.x0),
        "boundary": float(args.boundary),
        "num_trials": int(args.num_trials),
        "seed": int(args.seed),
        "chunk_ms": int(args.chunk_ms),
        "resume": bool(args.resume),
        "save_traj": bool(args.save_traj),
        "summary_file": str(summary_path.name),
        "result_dir": str(results_dir.name),
    }
    if calibration is not None:
        config_payload["calibration"] = {
            "kappa": float(calibration["kappa"]),
            "certificate_passed": bool(calibration["certificate_passed"]),
            "c_be_theta_max": float(calibration["c_be_theta_max"]),
        }
    config_path.write_text(json.dumps(config_payload, indent=2))

    print("Psychometric data summary")
    print(f"model: {args.model}")
    print("decision_mode: continuous")
    print(f"output_dir: {output_dir}")
    print(f"conditions: {len(summary_rows)}")
    print(f"summary_file: {summary_path}")
    print(f"max_miss_fraction: {max(float(row['miss_fraction']) for row in summary_rows):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
