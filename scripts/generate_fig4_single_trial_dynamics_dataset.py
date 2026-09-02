#!/usr/bin/env python3
"""Generate full single-condition neural dynamics for Figure 4 examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CANN_DDM_model_rate_based import CANN_DDM_model
from rate_model_core.accumulator_simulation import (
    _build_circuit_baseline_params,
    _calibration_metadata,
    _classify_hit_value,
    _classify_terminal_value,
)


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
DEFAULT_OUTPUT_ROOT = Path("results/figure4/fig4_single_trial_dynamics")
SELECTED_NEURON_START = 493
SELECTED_NEURON_STOP_EXCLUSIVE = 554
PARAM_CONFIG_TOP_LEVEL_KEYS = {
    "description",
    "edge_pop",
    "bump_pop",
    "decision_space_params",
    "geometry",
    "c_be_sweep",
    "selected_neuron_start",
    "selected_neuron_stop_exclusive",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _load_param_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    payload = json.loads(Path(config_path).read_text())
    unknown = sorted(set(payload) - PARAM_CONFIG_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"Unknown param-config keys: {unknown}")
    return payload


def _apply_param_overrides(params: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for section in ("edge_pop", "bump_pop", "decision_space_params", "geometry"):
        section_overrides = overrides.get(section)
        if section_overrides is None:
            continue
        if not isinstance(section_overrides, dict):
            raise TypeError(f"param-config section '{section}' must be a JSON object")
        params.setdefault(section, {}).update(section_overrides)
    return params


def _build_fig4_params(
    *,
    coherence: float,
    noise_scale: float,
    seed: int,
    dur: int,
    kappa: float | None,
    param_overrides: dict[str, Any],
) -> dict[str, Any]:
    params = _build_circuit_baseline_params(
        coherence=float(coherence),
        drift_gain=FIXED_DRIFT_GAIN,
        noise_scale=float(noise_scale),
        dt_ddm=FIXED_DT_DDM,
        t_start=FIXED_T_START,
        dur=int(dur),
        max_time=int(dur),
        seed=int(seed),
        kappa=None if kappa is None else float(kappa),
        theta_margin=0.02,
        x0=FIXED_X0,
        boundary=FIXED_BOUNDARY,
        mar=FIXED_MAR,
        c_eb=FIXED_C_EB,
    )
    return _apply_param_overrides(params, param_overrides)


def _prepare_fig4_target_diffusion_calibration(
    *,
    noise_scale: float,
    dur: int,
    seed: int,
    param_overrides: dict[str, Any],
) -> dict[str, Any]:
    model = CANN_DDM_model(
        CANN_params=_build_fig4_params(
            coherence=0.0,
            noise_scale=float(noise_scale),
            seed=int(seed),
            dur=int(dur),
            kappa=None,
            param_overrides=param_overrides,
        )
    )
    c_be_sweep = param_overrides.get("c_be_sweep")
    if c_be_sweep is not None:
        c_be_sweep = np.asarray(c_be_sweep, dtype=float)
    return model.prepare_target_diffusion_mode(c_be_sweep=c_be_sweep)


def _selected_neuron_indices(param_overrides: dict[str, Any]) -> np.ndarray:
    start = int(param_overrides.get("selected_neuron_start", SELECTED_NEURON_START))
    stop = int(param_overrides.get("selected_neuron_stop_exclusive", SELECTED_NEURON_STOP_EXCLUSIVE))
    if stop <= start:
        raise ValueError("selected_neuron_stop_exclusive must be greater than selected_neuron_start")
    return np.arange(start, stop, dtype=int)


def _trial_summary_row(
    *,
    trial_id: int,
    seed: int,
    coherence: float,
    choice: int,
    hit: bool,
    rt_ms: float,
    final_x: float,
    x_e: np.ndarray,
    r_e: np.ndarray,
    r_b: np.ndarray,
) -> dict[str, object]:
    return {
        "trial_id": int(trial_id),
        "seed": int(seed),
        "coherence": float(coherence),
        "choice": int(choice),
        "hit_boundary": bool(hit),
        "rt_ms": float(rt_ms),
        "final_x": float(final_x),
        "max_x_E": float(np.nanmax(x_e)),
        "min_x_E": float(np.nanmin(x_e)),
        "r_E_peak": float(np.nanmax(r_e)),
        "r_B_peak": float(np.nanmax(r_b)),
    }


def generate_dataset(
    *,
    coherence: float,
    noise_scale: float,
    num_trials: int,
    dur: int,
    seed: int,
    output_dir: Path,
    param_config: Path | None,
    batch_index: int,
    direct_trial_seeds: list[int] | None = None,
    dataset_filename: str = "single_condition_dynamics.npz",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / dataset_filename
    summary_path = output_dir / "trial_summary.csv"
    config_path = output_dir / "config.json"
    param_overrides = _load_param_config(param_config)

    dt_model_ms = int(round(FIXED_DT_MODEL))
    total_steps = int(int(dur) // dt_model_ms)
    time_ms = np.arange(total_steps, dtype=float) * float(dt_model_ms)

    calibration = _prepare_fig4_target_diffusion_calibration(
        noise_scale=float(noise_scale),
        dur=int(dur),
        seed=int(seed),
        param_overrides=param_overrides,
    )

    if direct_trial_seeds is None:
        seed_sequence = np.random.SeedSequence(int(seed))
        trial_sequences = seed_sequence.spawn(int(num_trials))
        trial_seeds = np.asarray([int(seq.generate_state(1)[0]) for seq in trial_sequences], dtype=np.uint32)
    else:
        if not direct_trial_seeds:
            raise ValueError("--trial-seeds must include at least one seed")
        trial_seeds = np.asarray([int(value) for value in direct_trial_seeds], dtype=np.uint32)
        num_trials = int(trial_seeds.size)

    probe_params = _build_fig4_params(
        coherence=float(coherence),
        noise_scale=float(noise_scale),
        dur=int(dur),
        seed=int(trial_seeds[0]),
        kappa=float(calibration["kappa"]),
        param_overrides=param_overrides,
    )
    probe_model = CANN_DDM_model(CANN_params=probe_params)
    num_units = int(probe_model.num_E)
    selected_neuron_indices = _selected_neuron_indices(param_overrides)
    if selected_neuron_indices[-1] >= num_units:
        raise ValueError(
            f"selected neuron index {int(selected_neuron_indices[-1])} exceeds "
            f"available units {num_units}"
        )
    num_selected_units = int(selected_neuron_indices.size)

    x_e = np.zeros((int(num_trials), total_steps), dtype=np.float32)
    x_b = np.zeros((int(num_trials), total_steps), dtype=np.float32)
    x_true = np.zeros((int(num_trials), total_steps), dtype=np.float32)
    r_e = np.zeros((int(num_trials), total_steps, num_selected_units), dtype=np.float32)
    r_b = np.zeros((int(num_trials), total_steps, num_selected_units), dtype=np.float32)
    choice = np.zeros(int(num_trials), dtype=np.int8)
    hit_boundary = np.zeros(int(num_trials), dtype=bool)
    rt_ms = np.full(int(num_trials), np.nan, dtype=np.float32)
    final_x = np.zeros(int(num_trials), dtype=np.float32)
    coherence_by_trial = np.full(int(num_trials), float(coherence), dtype=np.float32)

    summary_rows: list[dict[str, object]] = []
    start_time = time.time()
    for trial_id, trial_seed in enumerate(trial_seeds):
        print(
            f"[fig4-single-trial] trial={trial_id + 1}/{int(num_trials)} seed={int(trial_seed)}",
            file=sys.stderr,
            flush=True,
        )
        model = CANN_DDM_model(
            CANN_params=_build_fig4_params(
                coherence=float(coherence),
                noise_scale=float(noise_scale),
                dur=int(dur),
                seed=int(trial_seed),
                kappa=float(calibration["kappa"]),
                param_overrides=param_overrides,
            )
        )
        runner = model.run_simulation(
            mon_vars=["x_E", "x_B", "r_E", "r_B", "hit_boundary"],
            progress_bar=False,
            dt=FIXED_DT_MODEL,
            get_RT=False,
        )

        trial_x = np.asarray(runner.mon.x_E, dtype=np.float32).reshape(-1)[:total_steps]
        trial_x_b = np.asarray(runner.mon.x_B, dtype=np.float32).reshape(-1)[:total_steps]
        trial_x_true = np.asarray(model.x_traj, dtype=np.float32).reshape(-1)[:total_steps]
        trial_hit_trace = np.asarray(runner.mon.hit_boundary).reshape(-1).astype(bool)[:total_steps]
        trial_r_e_full = np.asarray(runner.mon.r_E, dtype=np.float32)[:total_steps]
        trial_r_b_full = np.asarray(runner.mon.r_B, dtype=np.float32)[:total_steps]
        trial_r_e = trial_r_e_full[:, selected_neuron_indices]
        trial_r_b = trial_r_b_full[:, selected_neuron_indices]

        x_e[trial_id] = trial_x
        x_b[trial_id] = trial_x_b
        x_true[trial_id] = trial_x_true
        r_e[trial_id] = trial_r_e
        r_b[trial_id] = trial_r_b

        hit_indices = np.flatnonzero(trial_hit_trace)
        trial_hit = bool(hit_indices.size > 0)
        if trial_hit:
            hit_index = int(hit_indices[0])
            hit_x = float(trial_x[hit_index])
            trial_choice = int(_classify_hit_value(hit_x, boundary=FIXED_BOUNDARY))
            trial_rt = float(hit_index * FIXED_DT_MODEL - FIXED_T_START)
            trial_final_x = float(FIXED_BOUNDARY) if trial_choice == 1 else 0.0
        else:
            trial_choice = int(_classify_terminal_value(float(trial_x[-1]), boundary=FIXED_BOUNDARY))
            trial_rt = float("nan")
            trial_final_x = float(trial_x[-1])

        choice[trial_id] = int(trial_choice)
        hit_boundary[trial_id] = bool(trial_hit)
        rt_ms[trial_id] = float(trial_rt)
        final_x[trial_id] = float(trial_final_x)
        summary_rows.append(
            _trial_summary_row(
                trial_id=trial_id,
                seed=int(trial_seed),
                coherence=float(coherence),
                choice=trial_choice,
                hit=trial_hit,
                rt_ms=trial_rt,
                final_x=trial_final_x,
                x_e=trial_x,
                r_e=trial_r_e,
                r_b=trial_r_b,
            )
        )

    metadata = {
        "dataset": "fig4_single_trial_dynamics",
        "coherence": float(coherence),
        "drift_gain": float(FIXED_DRIFT_GAIN),
        "drift_rate": float(FIXED_DRIFT_GAIN * float(coherence)),
        "noise_scale": float(noise_scale),
        "dt_ddm": float(FIXED_DT_DDM),
        "dt_model": float(FIXED_DT_MODEL),
        "t_start": int(FIXED_T_START),
        "dur": int(dur),
        "max_time": int(dur),
        "num_trials": int(num_trials),
        "batch_index": int(batch_index),
        "num_units": int(num_units),
        "num_selected_units": int(num_selected_units),
        "selected_neuron_indices": selected_neuron_indices.tolist(),
        "selected_neuron_index_start": int(selected_neuron_indices[0]),
        "selected_neuron_index_stop_exclusive": int(selected_neuron_indices[-1]) + 1,
        "neuron_subset": (
            f"global_indices_{int(selected_neuron_indices[0])}_to_"
            f"{int(selected_neuron_indices[-1])}_inclusive"
        ),
        "seed": int(seed),
        "trial_seeds": trial_seeds.tolist(),
        "direct_trial_seeds": None if direct_trial_seeds is None else trial_seeds.tolist(),
        "boundary": float(FIXED_BOUNDARY),
        "mar": float(FIXED_MAR),
        "c_eb": float(probe_model.c_EB),
        "x0": float(FIXED_X0),
        "param_config_path": None if param_config is None else str(Path(param_config)),
        "param_overrides": param_overrides,
        "effective_edge_pop": _json_ready(probe_params["edge_pop"]),
        "effective_bump_pop": _json_ready(probe_params["bump_pop"]),
        "effective_decision_space_params": _json_ready(probe_params["decision_space_params"]),
        "effective_geometry": _json_ready(probe_params["geometry"]),
        "mon_vars": ["x_E", "x_B", "r_E", "r_B", "hit_boundary"],
        "dataset_file": dataset_path.name,
        "summary_file": summary_path.name,
        "config_file": config_path.name,
        "elapsed_sec": float(time.time() - start_time),
        **_calibration_metadata(calibration),
    }

    np.savez_compressed(
        dataset_path,
        coherence_values=np.asarray([float(coherence)], dtype=float),
        choice=choice,
        hit_boundary=hit_boundary,
        rt_ms=rt_ms,
        final_x=final_x,
        coherence_by_trial=coherence_by_trial,
        time_ms=time_ms,
        x_E=x_e,
        x_B=x_b,
        x_true=x_true,
        r_E=r_e,
        r_B=r_b,
        metadata_json=np.asarray(json.dumps(_json_ready(metadata))),
    )

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    config_path.write_text(json.dumps(_json_ready(metadata), indent=2))

    hit_mask = hit_boundary & np.isfinite(rt_ms)
    correct_mask = hit_mask & (choice == 1)
    print(f"saved_dataset {dataset_path}")
    print(f"saved_summary {summary_path}")
    print(f"saved_config {config_path}")
    print(f"num_trials {int(num_trials)}")
    print(f"num_hit {int(hit_mask.sum())}")
    print(f"num_right_choice {int(np.sum(choice == 1))}")
    print(f"num_correct_right_hit {int(correct_mask.sum())}")
    if np.any(hit_mask):
        print(f"mean_rt_ms {float(np.nanmean(rt_ms[hit_mask])):.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coherence", type=float, default=0.5)
    parser.add_argument("--noise-scale", type=float, default=FIXED_NOISE_SCALE)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--dur", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-filename", type=str, default="single_condition_dynamics.npz")
    parser.add_argument("--param-config", type=Path, default=None)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument(
        "--trial-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Use these seeds directly as trial seeds instead of spawning seeds "
            "from --seed. This is useful for reproducing a specific saved trial."
        ),
    )
    args = parser.parse_args(argv)

    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if args.dur <= 0:
        raise ValueError("--dur must be positive")
    if args.noise_scale < 0:
        raise ValueError("--noise-scale must be non-negative")

    generate_dataset(
        coherence=float(args.coherence),
        noise_scale=float(args.noise_scale),
        num_trials=int(args.num_trials),
        dur=int(args.dur),
        seed=int(args.seed),
        output_dir=args.output_dir,
        param_config=args.param_config,
        batch_index=int(args.batch_index),
        direct_trial_seeds=args.trial_seeds,
        dataset_filename=str(args.dataset_filename),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
