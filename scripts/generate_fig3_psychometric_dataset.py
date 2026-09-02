#!/usr/bin/env python3
"""Generate Figure 3 psychometric datasets for DDM and circuit models."""

from __future__ import annotations

import argparse
import csv
import json
import time
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rate_model_core.accumulator_simulation import (
    AccumulatorSimulationSweep,
    ContinuousDDMConfig,
    load_simulation_sweep_npz,
    prepare_circuit_target_diffusion_calibration,
    save_simulation_sweep_npz,
    simulate_circuit_trials,
    simulate_ddm_trials,
)


DEFAULT_CONDITIONS = "-1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0"
DEFAULT_DATASET_NAME = "dataset.npz"
DEFAULT_SUMMARY_NAME = "summary.csv"
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_BATCH_TRIALS = 1000
FIXED_SEED = 201
FIXED_NOISE_SCALE = 0.5
FIXED_DT_DDM = 5.0
FIXED_DT_MODEL = 1.0
FIXED_T_START = 10
FIXED_X0 = 0.5
FIXED_BOUNDARY = 1.0
FIXED_MAR = 0.01
FIXED_C_EB = 0.1
FIXED_DRIFT_GAIN = 1.0
FIXED_DECISION_PARADIGM = "free_response"
FIXED_SAVE_TRAJ = False
DEFAULT_CALIBRATION_NAME = "shared_calibration.json"
R_E_METADATA_KEYS = (
    "selected_edge_indices",
    "selection_x_start",
    "selection_x_end",
    "selection_r_start_max",
    "selection_r_end_min",
    "selection_start_pos",
    "selection_end_pos",
    "num_selected_edge_units",
)


def parse_conditions(raw: str) -> np.ndarray:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("conditions must contain at least one numeric drift rate")
    return np.asarray(values, dtype=float)


def drift_label(drift_rate: float) -> str:
    sign = "p" if float(drift_rate) >= 0.0 else "m"
    magnitude = f"{abs(float(drift_rate)):.6f}".rstrip("0").rstrip(".") or "0"
    return f"drift_{sign}{magnitude.replace('.', 'p')}"


def binomial_half_width(p_right: float, num_hit: int) -> float:
    if int(num_hit) <= 0 or not np.isfinite(float(p_right)):
        return float("nan")
    return 1.96 * np.sqrt(float(p_right) * (1.0 - float(p_right)) / float(num_hit))


def summarize_condition(
    *,
    model: str,
    condition_label: str,
    drift_rate: float,
    noise_scale: float,
    num_trials: int,
    seed: int,
    dataset_name: str,
    result,
) -> dict[str, object]:
    hit_mask = np.asarray(result.hit_boundary, dtype=bool)
    num_hit = int(np.sum(hit_mask))
    p_right = float(np.mean(np.asarray(result.choice)[hit_mask] == 1)) if num_hit > 0 else float("nan")
    hit_fraction = float(np.mean(hit_mask))
    mean_rt_ms = float(np.nanmean(np.asarray(result.rt_ms, dtype=float))) if num_hit > 0 else float("nan")
    return {
        "model": str(model),
        "decision_mode": "continuous",
        "condition_label": str(condition_label),
        "coherence": float(drift_rate),
        "drift_rate": float(drift_rate),
        "noise_scale": float(noise_scale),
        "p_right": p_right,
        "num_hit": num_hit,
        "miss_fraction": 1.0 - hit_fraction,
        "hit_fraction": hit_fraction,
        "mean_rt_ms": mean_rt_ms,
        "ci_half_width": binomial_half_width(p_right, num_hit),
        "num_trials": int(num_trials),
        "seed": int(seed),
        "result_file": str(dataset_name),
    }


def stack_time_ms(*time_axes: np.ndarray) -> np.ndarray:
    arrays = [np.asarray(axis, dtype=float) for axis in time_axes]
    max_len = max(array.shape[0] for array in arrays)
    stacked = np.full((len(arrays), max_len), np.nan, dtype=float)
    for index, array in enumerate(arrays):
        stacked[index, : array.shape[0]] = array
    return stacked


def extract_r_e_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: metadata[key] for key in R_E_METADATA_KEYS if key in metadata}


def compute_shared_calibration(*, dur: int, seed: int = FIXED_SEED) -> dict[str, Any]:
    calibration = prepare_circuit_target_diffusion_calibration(
        dt_ddm=FIXED_DT_DDM,
        t_start=FIXED_T_START,
        dur=int(dur),
        max_time=int(dur),
        seed=int(seed),
        mar=FIXED_MAR,
        c_eb=FIXED_C_EB,
    )
    return {
        "model_type": "circuit_shared_calibration",
        "calibration_mode": "target_diffusion",
        "condition_axis": "drift_rate",
        "calibration_condition": {
            "coherence": 0.0,
            "drift_gain": 0.0,
            "drift_rate": 0.0,
            "noise_scale": 0.0,
        },
        "dt_ddm": FIXED_DT_DDM,
        "dt_model": FIXED_DT_MODEL,
        "t_start": FIXED_T_START,
        "dur": int(dur),
        "max_time": int(dur),
        "x0": FIXED_X0,
        "boundary": FIXED_BOUNDARY,
        "mar": FIXED_MAR,
        "c_eb": FIXED_C_EB,
        "seed": int(seed),
        "theta_margin": 0.02,
        "kappa": float(calibration["kappa"]),
        "certificate_passed": bool(calibration["certificate_passed"]),
        "c_be_theta_max": float(calibration["c_be_theta_max"]),
        "effective_c_be_max": float(calibration["effective_c_be_max"]),
        "valid_c_be_max": float(calibration["valid_c_be_max"]),
        "max_abs_v_drive": float(calibration["max_abs_v_drive"]),
        "kappa_rel_error": float(calibration["kappa_rel_error"]),
    }


def write_shared_calibration(*, output_path: Path, dur: int, seed: int = FIXED_SEED) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = compute_shared_calibration(dur=int(dur), seed=int(seed))
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def load_shared_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "kappa": float(payload["kappa"]),
        "certificate_passed": bool(payload["certificate_passed"]),
        "c_be_theta_max": float(payload["c_be_theta_max"]),
    }


def batch_label(batch_index: int) -> str:
    return f"batch_{int(batch_index):03d}"


def split_batch_trial_counts(num_trials: int, batch_trials: int = DEFAULT_BATCH_TRIALS) -> list[int]:
    total = int(num_trials)
    chunk = int(batch_trials)
    if total <= 0:
        raise ValueError("num_trials must be positive")
    if chunk <= 0:
        raise ValueError("batch_trials must be positive")
    full_batches, remainder = divmod(total, chunk)
    counts = [chunk for _ in range(full_batches)]
    if remainder:
        counts.append(remainder)
    return counts


def prepare_run_root(*, run_root: Path, dur: int, seed: int = FIXED_SEED) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "conditions").mkdir(parents=True, exist_ok=True)
    calibration_path = run_root / DEFAULT_CALIBRATION_NAME
    if not calibration_path.exists():
        write_shared_calibration(output_path=calibration_path, dur=int(dur), seed=int(seed))
    return calibration_path


def run_single_condition(
    *,
    drift_rate: float,
    num_trials: int,
    dur: int,
    output_dir: Path,
    condition_index: int,
    calibration_file: Path | None = None,
    batch_index: int = 0,
    save_r_e: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = DEFAULT_DATASET_NAME
    summary_name = DEFAULT_SUMMARY_NAME
    config_name = DEFAULT_CONFIG_NAME
    label = drift_label(float(drift_rate))
    start_time = time.time()
    print(
        f"[fig3-dataset] start condition={label} coherence={float(drift_rate):g} num_trials={int(num_trials)} save_r_e={bool(save_r_e)}",
        file=sys.stderr,
        flush=True,
    )

    ddm_seed_seq, circuit_seed_seq = np.random.SeedSequence(FIXED_SEED).spawn(2)
    ddm_condition_seq = ddm_seed_seq.spawn(condition_index + 1)[condition_index]
    circuit_condition_seq = circuit_seed_seq.spawn(condition_index + 1)[condition_index]
    ddm_seed = int(ddm_condition_seq.spawn(batch_index + 1)[batch_index].generate_state(1)[0])
    circuit_seed = int(circuit_condition_seq.spawn(batch_index + 1)[batch_index].generate_state(1)[0])

    ddm_result = simulate_ddm_trials(
        decision_mode="continuous",
        config=ContinuousDDMConfig(
            drift_rate=float(drift_rate),
            noise_scale=FIXED_NOISE_SCALE,
        ),
        dt_DDM=FIXED_DT_DDM,
        dur=int(dur),
        max_time=int(dur),
        t_start=FIXED_T_START,
        x0=FIXED_X0,
        boundary=FIXED_BOUNDARY,
        num_trials=int(num_trials),
        decision_paradigm=FIXED_DECISION_PARADIGM,
        seed=ddm_seed,
        return_traj=FIXED_SAVE_TRAJ,
    )

    calibration = (
        load_shared_calibration(Path(calibration_file))
        if calibration_file is not None
        else prepare_circuit_target_diffusion_calibration(
            dt_ddm=FIXED_DT_DDM,
            t_start=FIXED_T_START,
            dur=int(dur),
            max_time=int(dur),
            seed=FIXED_SEED,
            mar=FIXED_MAR,
            c_eb=FIXED_C_EB,
        )
    )
    circuit_result = simulate_circuit_trials(
        coherence=float(drift_rate),
        drift_gain=FIXED_DRIFT_GAIN,
        noise_scale=FIXED_NOISE_SCALE,
        dt_ddm=FIXED_DT_DDM,
        dt_model=FIXED_DT_MODEL,
        t_start=FIXED_T_START,
        dur=int(dur),
        max_time=int(dur),
        num_trials=int(num_trials),
        seed=int(circuit_seed),
        decision_paradigm=FIXED_DECISION_PARADIGM,
        save_traj=FIXED_SAVE_TRAJ,
        calibration=calibration,
        x0=FIXED_X0,
        boundary=FIXED_BOUNDARY,
        mar=FIXED_MAR,
        c_eb=FIXED_C_EB,
        chunk_ms=1000,
        save_r_e=bool(save_r_e),
        progress_every=max(1, int(num_trials) // 10),
    )

    ddm_sweep = AccumulatorSimulationSweep(
        coherence_values=np.asarray([float(drift_rate)], dtype=float),
        choice=np.asarray([np.asarray(ddm_result.choice)]),
        hit_boundary=np.asarray([np.asarray(ddm_result.hit_boundary)]),
        rt_ms=np.asarray([np.asarray(ddm_result.rt_ms, dtype=float)]),
        final_x=np.asarray([np.asarray(ddm_result.final_x, dtype=float)]),
        time_ms=np.asarray(ddm_result.time_ms, dtype=float),
        x_traj=None,
        metadata={
            "model_type": "ddm",
            "decision_mode": "continuous",
            "decision_paradigm": FIXED_DECISION_PARADIGM,
            "condition_axis": "drift_rate",
            "condition_labels": [label],
            "condition_drift_rates": [float(drift_rate)],
            "condition_noise_scales": [FIXED_NOISE_SCALE],
            "condition_num_trials": [int(num_trials)],
            "num_conditions": 1,
            "num_trials_per_condition": [int(num_trials)],
            "total_trials": int(num_trials),
            "dataset_file": dataset_name,
            "summary_file": summary_name,
            "config_file": config_name,
            "condition_seeds": [ddm_seed],
            "condition_seed": int(ddm_condition_seq.generate_state(1)[0]),
            "batch_index": int(batch_index),
        },
    )
    circuit_sweep = AccumulatorSimulationSweep(
        coherence_values=np.asarray([float(drift_rate)], dtype=float),
        choice=np.asarray([np.asarray(circuit_result.choice)]),
        hit_boundary=np.asarray([np.asarray(circuit_result.hit_boundary)]),
        rt_ms=np.asarray([np.asarray(circuit_result.rt_ms, dtype=float)]),
        final_x=np.asarray([np.asarray(circuit_result.final_x, dtype=float)]),
        time_ms=np.asarray(circuit_result.time_ms, dtype=float),
        x_traj=None,
        r_e=(
            np.asarray([np.asarray(circuit_result.r_e, dtype=np.float32)], dtype=np.float32)
            if circuit_result.r_e is not None
            else None
        ),
        metadata={
            "model_type": "circuit",
            "decision_mode": "continuous",
            "decision_paradigm": FIXED_DECISION_PARADIGM,
            "condition_axis": "drift_rate",
            "condition_labels": [label],
            "condition_coherence_values": [float(drift_rate)],
            "condition_drift_rates": [float(drift_rate)],
            "condition_noise_scales": [FIXED_NOISE_SCALE],
            "condition_num_trials": [int(num_trials)],
            "num_conditions": 1,
            "num_trials_per_condition": [int(num_trials)],
            "total_trials": int(num_trials),
            "dataset_file": dataset_name,
            "summary_file": summary_name,
            "config_file": config_name,
            "condition_seeds": [circuit_seed],
            "condition_seed": int(circuit_condition_seq.generate_state(1)[0]),
            "batch_index": int(batch_index),
            "calibration": {
                "kappa": float(calibration["kappa"]),
                "certificate_passed": bool(calibration["certificate_passed"]),
                "c_be_theta_max": float(calibration["c_be_theta_max"]),
            },
            "save_r_e": bool(save_r_e),
            **extract_r_e_metadata(circuit_result.metadata),
        },
    )

    payload = {
        "model_names": np.asarray(["ddm", "circuit"]),
        "coherence_values": np.asarray([float(drift_rate)], dtype=float),
        "choice": np.stack([ddm_sweep.choice, circuit_sweep.choice], axis=0),
        "hit_boundary": np.stack([ddm_sweep.hit_boundary, circuit_sweep.hit_boundary], axis=0),
        "rt_ms": np.stack([ddm_sweep.rt_ms, circuit_sweep.rt_ms], axis=0),
        "final_x": np.stack([ddm_sweep.final_x, circuit_sweep.final_x], axis=0),
        "time_ms": stack_time_ms(ddm_sweep.time_ms, circuit_sweep.time_ms),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "dataset_file": dataset_name,
                    "summary_file": summary_name,
                    "config_file": config_name,
                    "condition_axis": "drift_rate",
                    "num_models": 2,
                    "num_conditions": 1,
                    "num_trials": int(num_trials),
                    "ddm_metadata": ddm_sweep.metadata,
                    "circuit_metadata": circuit_sweep.metadata,
                }
            )
        ),
    }
    np.savez_compressed(output_dir / dataset_name, **payload)
    save_simulation_sweep_npz(output_dir / "ddm_dataset.npz", ddm_sweep)
    save_simulation_sweep_npz(output_dir / "circuit_dataset.npz", circuit_sweep)

    summary_rows = [
        summarize_condition(
            model="ddm",
            condition_label=label,
            drift_rate=float(drift_rate),
            noise_scale=FIXED_NOISE_SCALE,
            num_trials=int(num_trials),
            seed=ddm_seed,
            dataset_name=dataset_name,
            result=ddm_result,
        ),
        summarize_condition(
            model="circuit",
            condition_label=label,
            drift_rate=float(drift_rate),
            noise_scale=FIXED_NOISE_SCALE,
            num_trials=int(num_trials),
            seed=circuit_seed,
            dataset_name=dataset_name,
            result=circuit_result,
        ),
    ]
    summary_rows.sort(key=lambda row: str(row["model"]))
    with (output_dir / summary_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    config = {
        "models": ["ddm", "circuit"],
        "condition_axis": "drift_rate",
        "conditions": [
            {
                "label": label,
                "coherence": float(drift_rate),
                "drift_rate": float(drift_rate),
                "noise_scale": FIXED_NOISE_SCALE,
            }
        ],
        "coherence_values": [float(drift_rate)],
        "drift_gain": FIXED_DRIFT_GAIN,
        "noise_scale": FIXED_NOISE_SCALE,
        "dt_ddm": FIXED_DT_DDM,
        "dt_model": FIXED_DT_MODEL,
        "t_start": FIXED_T_START,
        "dur": int(dur),
        "max_time": int(dur),
        "x0": FIXED_X0,
        "boundary": FIXED_BOUNDARY,
        "mar": FIXED_MAR,
        "c_eb": FIXED_C_EB,
        "num_trials": int(num_trials),
        "seed": FIXED_SEED,
        "save_traj": FIXED_SAVE_TRAJ,
        "save_r_e": bool(save_r_e),
        **extract_r_e_metadata(circuit_result.metadata),
        "dataset_file": dataset_name,
        "summary_file": summary_name,
        "num_models": 2,
        "num_conditions": 1,
        "condition_index": int(condition_index),
        "batch_index": int(batch_index),
    }
    (output_dir / config_name).write_text(json.dumps(config, indent=2))
    elapsed = time.time() - start_time
    print(
        f"[fig3-dataset] done condition={label} elapsed_sec={elapsed:.1f}",
        file=sys.stderr,
        flush=True,
    )


def combine_condition_batches(
    *,
    batch_dirs: list[Path],
    drift_rate: float,
    num_trials: int,
) -> tuple[AccumulatorSimulationSweep, AccumulatorSimulationSweep, list[dict[str, object]]]:
    if not batch_dirs:
        raise ValueError("batch_dirs must not be empty")

    ddm_sweeps = [load_simulation_sweep_npz(batch_dir / "ddm_dataset.npz") for batch_dir in batch_dirs]
    circuit_sweeps = [load_simulation_sweep_npz(batch_dir / "circuit_dataset.npz") for batch_dir in batch_dirs]

    ddm_choice = np.concatenate([np.asarray(sweep.choice) for sweep in ddm_sweeps], axis=1)
    ddm_hit = np.concatenate([np.asarray(sweep.hit_boundary) for sweep in ddm_sweeps], axis=1)
    ddm_rt = np.concatenate([np.asarray(sweep.rt_ms, dtype=float) for sweep in ddm_sweeps], axis=1)
    ddm_final_x = np.concatenate([np.asarray(sweep.final_x, dtype=float) for sweep in ddm_sweeps], axis=1)
    circuit_choice = np.concatenate([np.asarray(sweep.choice) for sweep in circuit_sweeps], axis=1)
    circuit_hit = np.concatenate([np.asarray(sweep.hit_boundary) for sweep in circuit_sweeps], axis=1)
    circuit_rt = np.concatenate([np.asarray(sweep.rt_ms, dtype=float) for sweep in circuit_sweeps], axis=1)
    circuit_final_x = np.concatenate([np.asarray(sweep.final_x, dtype=float) for sweep in circuit_sweeps], axis=1)
    have_r_e = circuit_sweeps[0].r_e is not None
    if any((sweep.r_e is not None) != have_r_e for sweep in circuit_sweeps):
        raise ValueError("All circuit batch sweeps must either all include r_e or all omit it")
    circuit_r_e = (
        np.concatenate([np.asarray(sweep.r_e, dtype=np.float32) for sweep in circuit_sweeps], axis=1)
        if have_r_e
        else None
    )

    label = drift_label(float(drift_rate))
    ddm_condition_seed = int(ddm_sweeps[0].metadata.get("condition_seed", ddm_sweeps[0].metadata["condition_seeds"][0]))
    circuit_condition_seed = int(circuit_sweeps[0].metadata.get("condition_seed", circuit_sweeps[0].metadata["condition_seeds"][0]))

    ddm_sweep = AccumulatorSimulationSweep(
        coherence_values=np.asarray([float(drift_rate)], dtype=float),
        choice=ddm_choice,
        hit_boundary=ddm_hit,
        rt_ms=ddm_rt,
        final_x=ddm_final_x,
        time_ms=np.asarray(ddm_sweeps[0].time_ms, dtype=float),
        x_traj=None,
        metadata={
            "model_type": "ddm",
            "decision_mode": "continuous",
            "decision_paradigm": FIXED_DECISION_PARADIGM,
            "condition_axis": "drift_rate",
            "condition_labels": [label],
            "condition_drift_rates": [float(drift_rate)],
            "condition_noise_scales": [FIXED_NOISE_SCALE],
            "condition_num_trials": [int(num_trials)],
            "num_conditions": 1,
            "num_trials_per_condition": [int(num_trials)],
            "total_trials": int(num_trials),
            "dataset_file": DEFAULT_DATASET_NAME,
            "summary_file": DEFAULT_SUMMARY_NAME,
            "config_file": DEFAULT_CONFIG_NAME,
            "condition_seeds": [ddm_condition_seed],
            "condition_seed": ddm_condition_seed,
            "batch_seeds": [int(sweep.metadata["condition_seeds"][0]) for sweep in ddm_sweeps],
        },
    )
    circuit_sweep = AccumulatorSimulationSweep(
        coherence_values=np.asarray([float(drift_rate)], dtype=float),
        choice=circuit_choice,
        hit_boundary=circuit_hit,
        rt_ms=circuit_rt,
        final_x=circuit_final_x,
        time_ms=np.asarray(circuit_sweeps[0].time_ms, dtype=float),
        x_traj=None,
        r_e=circuit_r_e,
        metadata={
            "model_type": "circuit",
            "decision_mode": "continuous",
            "decision_paradigm": FIXED_DECISION_PARADIGM,
            "condition_axis": "drift_rate",
            "condition_labels": [label],
            "condition_coherence_values": [float(drift_rate)],
            "condition_drift_rates": [float(drift_rate)],
            "condition_noise_scales": [FIXED_NOISE_SCALE],
            "condition_num_trials": [int(num_trials)],
            "num_conditions": 1,
            "num_trials_per_condition": [int(num_trials)],
            "total_trials": int(num_trials),
            "dataset_file": DEFAULT_DATASET_NAME,
            "summary_file": DEFAULT_SUMMARY_NAME,
            "config_file": DEFAULT_CONFIG_NAME,
            "condition_seeds": [circuit_condition_seed],
            "condition_seed": circuit_condition_seed,
            "batch_seeds": [int(sweep.metadata["condition_seeds"][0]) for sweep in circuit_sweeps],
            "calibration": circuit_sweeps[0].metadata["calibration"],
            "save_r_e": bool(have_r_e),
            **extract_r_e_metadata(circuit_sweeps[0].metadata),
        },
    )

    ddm_result = type("MergedResult", (), {
        "choice": ddm_choice.reshape(-1),
        "hit_boundary": ddm_hit.reshape(-1),
        "rt_ms": ddm_rt.reshape(-1),
        "final_x": ddm_final_x.reshape(-1),
    })()
    circuit_result = type("MergedResult", (), {
        "choice": circuit_choice.reshape(-1),
        "hit_boundary": circuit_hit.reshape(-1),
        "rt_ms": circuit_rt.reshape(-1),
        "final_x": circuit_final_x.reshape(-1),
    })()
    summary_rows = [
        summarize_condition(
            model="ddm",
            condition_label=label,
            drift_rate=float(drift_rate),
            noise_scale=FIXED_NOISE_SCALE,
            num_trials=int(num_trials),
            seed=ddm_condition_seed,
            dataset_name=DEFAULT_DATASET_NAME,
            result=ddm_result,
        ),
        summarize_condition(
            model="circuit",
            condition_label=label,
            drift_rate=float(drift_rate),
            noise_scale=FIXED_NOISE_SCALE,
            num_trials=int(num_trials),
            seed=circuit_condition_seed,
            dataset_name=DEFAULT_DATASET_NAME,
            result=circuit_result,
        ),
    ]
    return ddm_sweep, circuit_sweep, summary_rows


def merge_condition_outputs(
    *,
    run_root: Path,
    drift_rates: np.ndarray,
    num_trials: int,
    dur: int,
) -> None:
    dataset_name = DEFAULT_DATASET_NAME
    summary_name = DEFAULT_SUMMARY_NAME
    config_name = DEFAULT_CONFIG_NAME
    ddm_sweeps = []
    circuit_sweeps = []
    summary_rows: list[dict[str, object]] = []
    for drift_rate in np.asarray(drift_rates, dtype=float):
        condition_dir = run_root / "conditions" / drift_label(float(drift_rate))
        if not condition_dir.exists():
            raise FileNotFoundError(f"Missing condition output directory: {condition_dir}")
        batch_dirs = sorted(
            path for path in condition_dir.iterdir()
            if path.is_dir() and (path / "ddm_dataset.npz").exists() and (path / "circuit_dataset.npz").exists()
        )
        if batch_dirs:
            ddm_sweep, circuit_sweep, condition_rows = combine_condition_batches(
                batch_dirs=batch_dirs,
                drift_rate=float(drift_rate),
                num_trials=int(num_trials),
            )
            ddm_sweeps.append(ddm_sweep)
            circuit_sweeps.append(circuit_sweep)
            summary_rows.extend(condition_rows)
            continue

        direct_ddm = condition_dir / "ddm_dataset.npz"
        direct_circuit = condition_dir / "circuit_dataset.npz"
        if direct_ddm.exists() and direct_circuit.exists():
            ddm_sweep = load_simulation_sweep_npz(direct_ddm)
            circuit_sweep = load_simulation_sweep_npz(direct_circuit)
            ddm_sweeps.append(ddm_sweep)
            circuit_sweeps.append(circuit_sweep)
            with (condition_dir / summary_name).open(newline="") as handle:
                summary_rows.extend(csv.DictReader(handle))
            continue
        raise FileNotFoundError(f"Missing condition outputs under {condition_dir}")

    ddm_choice = np.concatenate([np.asarray(sweep.choice) for sweep in ddm_sweeps], axis=0)
    ddm_hit = np.concatenate([np.asarray(sweep.hit_boundary) for sweep in ddm_sweeps], axis=0)
    ddm_rt = np.concatenate([np.asarray(sweep.rt_ms, dtype=float) for sweep in ddm_sweeps], axis=0)
    ddm_final_x = np.concatenate([np.asarray(sweep.final_x, dtype=float) for sweep in ddm_sweeps], axis=0)
    circuit_choice = np.concatenate([np.asarray(sweep.choice) for sweep in circuit_sweeps], axis=0)
    circuit_hit = np.concatenate([np.asarray(sweep.hit_boundary) for sweep in circuit_sweeps], axis=0)
    circuit_rt = np.concatenate([np.asarray(sweep.rt_ms, dtype=float) for sweep in circuit_sweeps], axis=0)
    circuit_final_x = np.concatenate([np.asarray(sweep.final_x, dtype=float) for sweep in circuit_sweeps], axis=0)
    have_r_e = bool(circuit_sweeps[0].metadata.get("save_r_e", False))

    combined_ddm_metadata = {
        "model_type": "ddm",
        "decision_mode": "continuous",
        "decision_paradigm": FIXED_DECISION_PARADIGM,
        "condition_axis": "drift_rate",
        "condition_labels": [drift_label(float(rate)) for rate in drift_rates],
        "condition_drift_rates": [float(rate) for rate in drift_rates],
        "condition_noise_scales": [FIXED_NOISE_SCALE for _ in drift_rates],
        "condition_num_trials": [int(num_trials) for _ in drift_rates],
        "num_conditions": int(len(drift_rates)),
        "num_trials_per_condition": [int(num_trials) for _ in drift_rates],
        "total_trials": int(len(drift_rates) * int(num_trials)),
        "dataset_file": dataset_name,
        "summary_file": summary_name,
        "config_file": config_name,
        "condition_seeds": [
            int(sweep.metadata["condition_seeds"][0]) for sweep in ddm_sweeps
        ],
    }
    combined_circuit_metadata = {
        "model_type": "circuit",
        "decision_mode": "continuous",
        "decision_paradigm": FIXED_DECISION_PARADIGM,
        "condition_axis": "drift_rate",
        "condition_labels": [drift_label(float(rate)) for rate in drift_rates],
        "condition_coherence_values": [float(rate) for rate in drift_rates],
        "condition_drift_rates": [float(rate) for rate in drift_rates],
        "condition_noise_scales": [FIXED_NOISE_SCALE for _ in drift_rates],
        "condition_num_trials": [int(num_trials) for _ in drift_rates],
        "num_conditions": int(len(drift_rates)),
        "num_trials_per_condition": [int(num_trials) for _ in drift_rates],
        "total_trials": int(len(drift_rates) * int(num_trials)),
        "dataset_file": dataset_name,
        "summary_file": summary_name,
        "config_file": config_name,
        "condition_seeds": [
            int(sweep.metadata["condition_seeds"][0]) for sweep in circuit_sweeps
        ],
        "calibration": circuit_sweeps[0].metadata["calibration"],
        "save_r_e": have_r_e,
        **extract_r_e_metadata(circuit_sweeps[0].metadata),
    }

    payload = {
        "model_names": np.asarray(["ddm", "circuit"]),
        "coherence_values": np.asarray(drift_rates, dtype=float),
        "choice": np.stack([ddm_choice, circuit_choice], axis=0),
        "hit_boundary": np.stack([ddm_hit, circuit_hit], axis=0),
        "rt_ms": np.stack([ddm_rt, circuit_rt], axis=0),
        "final_x": np.stack([ddm_final_x, circuit_final_x], axis=0),
        "time_ms": stack_time_ms(ddm_sweeps[0].time_ms, circuit_sweeps[0].time_ms),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "dataset_file": dataset_name,
                    "summary_file": summary_name,
                    "config_file": config_name,
                    "condition_axis": "drift_rate",
                    "num_models": 2,
                    "num_conditions": int(len(drift_rates)),
                    "num_trials": int(num_trials),
                    "ddm_metadata": combined_ddm_metadata,
                    "circuit_metadata": combined_circuit_metadata,
                }
            )
        ),
    }
    np.savez_compressed(run_root / dataset_name, **payload)

    summary_rows.sort(key=lambda row: (str(row["model"]), float(row["coherence"])))
    with (run_root / summary_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    config = {
        "models": ["ddm", "circuit"],
        "condition_axis": "drift_rate",
        "conditions": [
            {
                "label": drift_label(float(drift_rate)),
                "coherence": float(drift_rate),
                "drift_rate": float(drift_rate),
                "noise_scale": FIXED_NOISE_SCALE,
            }
            for drift_rate in np.asarray(drift_rates, dtype=float)
        ],
        "coherence_values": np.asarray(drift_rates, dtype=float).tolist(),
        "drift_gain": FIXED_DRIFT_GAIN,
        "noise_scale": FIXED_NOISE_SCALE,
        "dt_ddm": FIXED_DT_DDM,
        "dt_model": FIXED_DT_MODEL,
        "t_start": FIXED_T_START,
        "dur": int(dur),
        "max_time": int(dur),
        "x0": FIXED_X0,
        "boundary": FIXED_BOUNDARY,
        "mar": FIXED_MAR,
        "c_eb": FIXED_C_EB,
        "num_trials": int(num_trials),
        "seed": FIXED_SEED,
        "save_traj": FIXED_SAVE_TRAJ,
        "save_r_e": bool(have_r_e),
        **extract_r_e_metadata(circuit_sweeps[0].metadata),
        "dataset_file": dataset_name,
        "summary_file": summary_name,
        "num_models": 2,
        "num_conditions": int(len(drift_rates)),
    }
    (run_root / config_name).write_text(json.dumps(config, indent=2))


def run_all_conditions(
    *,
    run_name: str,
    output_root: Path,
    drift_rates: np.ndarray,
    num_trials: int,
    dur: int,
    save_r_e: bool = False,
) -> None:
    run_root = output_root / run_name
    for index, drift_rate in enumerate(np.asarray(drift_rates, dtype=float)):
        print(
            f"[fig3-dataset] condition_index={index + 1}/{len(np.asarray(drift_rates, dtype=float))}",
            file=sys.stderr,
            flush=True,
        )
        run_single_condition(
            drift_rate=float(drift_rate),
            num_trials=int(num_trials),
            dur=int(dur),
            output_dir=run_root / "conditions" / drift_label(float(drift_rate)),
            condition_index=index,
            save_r_e=bool(save_r_e),
        )
    merge_condition_outputs(
        run_root=run_root,
        drift_rates=np.asarray(drift_rates, dtype=float),
        num_trials=int(num_trials),
        dur=int(dur),
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--num-trials", type=int, default=1000)
    common.add_argument("--dur", type=int, default=4000)
    common.add_argument("--save-r-e", action="store_true")

    all_parser = subparsers.add_parser("all", parents=[common])
    all_parser.add_argument("--run-name", default="fig3_psychometric")
    all_parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    all_parser.add_argument("--output-root", type=Path, default=Path("results/figure3"))

    prepare_parser = subparsers.add_parser("prepare-run")
    prepare_parser.add_argument("--run-root", type=Path, required=True)
    prepare_parser.add_argument("--dur", type=int, default=4000)
    prepare_parser.add_argument("--seed", type=int, default=FIXED_SEED)

    single_parser = subparsers.add_parser("single-condition", parents=[common])
    single_parser.add_argument("--condition", type=float, required=True)
    single_parser.add_argument("--condition-index", type=int, required=True)
    single_parser.add_argument("--output-dir", type=Path, required=True)
    single_parser.add_argument("--calibration-file", type=Path)
    single_parser.add_argument("--batch-index", type=int, default=0)

    merge_parser = subparsers.add_parser("merge", parents=[common])
    merge_parser.add_argument("--run-root", type=Path, required=True)
    merge_parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--dur", type=int, default=4000)
    calibrate_parser.add_argument("--seed", type=int, default=FIXED_SEED)
    calibrate_parser.add_argument("--output", type=Path, default=Path("results/figure3") / DEFAULT_CALIBRATION_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    if args.command not in {"calibrate", "prepare-run"}:
        if int(args.num_trials) <= 0:
            raise ValueError("num-trials must be positive")
    if int(args.dur) <= 0:
        raise ValueError("dur must be positive")

    if args.command == "single-condition":
        run_single_condition(
            drift_rate=float(args.condition),
            num_trials=int(args.num_trials),
            dur=int(args.dur),
            output_dir=Path(args.output_dir),
            condition_index=int(args.condition_index),
            calibration_file=None if args.calibration_file is None else Path(args.calibration_file),
            batch_index=int(args.batch_index),
            save_r_e=bool(args.save_r_e),
        )
        return 0

    if args.command == "prepare-run":
        calibration_path = prepare_run_root(
            run_root=Path(args.run_root),
            dur=int(args.dur),
            seed=int(args.seed),
        )
        print("Figure 3 prepared run")
        print(f"run_root: {Path(args.run_root)}")
        print(f"calibration: {calibration_path}")
        return 0

    if args.command == "merge":
        merge_condition_outputs(
            run_root=Path(args.run_root),
            drift_rates=parse_conditions(args.conditions),
            num_trials=int(args.num_trials),
            dur=int(args.dur),
        )
        return 0

    if args.command == "all":
        run_all_conditions(
            run_name=str(args.run_name),
            output_root=Path(args.output_root),
            drift_rates=parse_conditions(args.conditions),
            num_trials=int(args.num_trials),
            dur=int(args.dur),
            save_r_e=bool(args.save_r_e),
        )
        return 0

    if args.command == "calibrate":
        output_path = write_shared_calibration(
            output_path=Path(args.output),
            dur=int(args.dur),
            seed=int(args.seed),
        )
        print("Figure 3 shared calibration")
        print(f"output: {output_path}")
        return 0

    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
