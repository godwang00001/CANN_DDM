"""Shared task-level accumulator simulation helpers for DDM and circuit runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import sys
from typing import Any

import numpy as np


THETA_MARGIN_DEFAULT = 0.02
C_BE_SWEEP_DEFAULT = np.array([0.0, 0.04], dtype=float)
MIN_ACCUMULATION_SAMPLES_DEFAULT = 40
X0_DEFAULT = 0.5
BOUNDARY_DEFAULT = 1.0
COMMIT_MARGIN_DEFAULT = 0.0
CHOICE_TOL = 1e-6
CIRCUIT_CHUNK_MS_DEFAULT = 1000
R_E_SELECTION_START_X_DEFAULT = 0.5
R_E_SELECTION_END_X_DEFAULT = 0.99
R_E_SELECTION_START_MAX_DEFAULT = 0.25
R_E_SELECTION_END_MIN_DEFAULT = 0.75


@dataclass(frozen=True)
class AccumulatorSimulationResult:
    """Container for a batch of task-level accumulator trials."""

    choice: np.ndarray
    hit_boundary: np.ndarray
    rt_ms: np.ndarray
    final_x: np.ndarray
    time_ms: np.ndarray
    x_traj: np.ndarray | None = None
    r_e: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccumulatorSimulationSweep:
    """Container for a psychometric sweep over multiple coherence conditions."""

    coherence_values: np.ndarray
    choice: np.ndarray
    hit_boundary: np.ndarray
    rt_ms: np.ndarray
    final_x: np.ndarray
    time_ms: np.ndarray
    x_traj: np.ndarray | None = None
    r_e: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContinuousDDMConfig:
    """Task-level configuration for the continuous absorbed DDM."""

    drift_rate: float
    noise_scale: float


@dataclass(frozen=True)
class PulseDDMConfig:
    """Task-level configuration for the pulse/click-based absorbed DDM."""

    lambda_click_L: float
    lambda_click_R: float
    delta_click_x: float
    noise_scale: float
    lambda_commit: float = 0.0


def decision_step_ms(dt_ms: float) -> int:
    """Validate and return the integer step size in milliseconds."""

    step_ms = int(round(float(dt_ms)))
    if step_ms <= 0 or not np.isclose(float(dt_ms), float(step_ms)):
        raise ValueError("step size must be a positive integer number of milliseconds")
    return step_ms


def _json_ready_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            ready[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer, np.bool_)):
            ready[key] = value.item()
        else:
            ready[key] = value
    return ready


def save_simulation_result_npz(path: str | Path, result: AccumulatorSimulationResult) -> Path:
    """Save a shared accumulator result to a compressed NumPy archive."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "choice": np.asarray(result.choice),
        "hit_boundary": np.asarray(result.hit_boundary),
        "rt_ms": np.asarray(result.rt_ms),
        "final_x": np.asarray(result.final_x),
        "time_ms": np.asarray(result.time_ms),
        "metadata_json": np.asarray(json.dumps(_json_ready_metadata(dict(result.metadata)))),
    }
    if result.x_traj is not None:
        payload["x_traj"] = np.asarray(result.x_traj)
    if result.r_e is not None:
        payload["r_e"] = np.asarray(result.r_e, dtype=np.float32)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_simulation_result_npz(path: str | Path) -> AccumulatorSimulationResult:
    """Load a shared accumulator result from a compressed NumPy archive."""

    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        x_traj = np.asarray(data["x_traj"]) if "x_traj" in data.files else None
        r_e = np.asarray(data["r_e"], dtype=np.float32) if "r_e" in data.files else None
        return AccumulatorSimulationResult(
            choice=np.asarray(data["choice"]),
            hit_boundary=np.asarray(data["hit_boundary"]),
            rt_ms=np.asarray(data["rt_ms"], dtype=float),
            final_x=np.asarray(data["final_x"], dtype=float),
            time_ms=np.asarray(data["time_ms"], dtype=float),
            x_traj=x_traj,
            r_e=r_e,
            metadata=metadata,
        )


def save_simulation_sweep_npz(path: str | Path, sweep: AccumulatorSimulationSweep) -> Path:
    """Save a coherence sweep to a compressed NumPy archive."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "coherence_values": np.asarray(sweep.coherence_values, dtype=float),
        "choice": np.asarray(sweep.choice),
        "hit_boundary": np.asarray(sweep.hit_boundary),
        "rt_ms": np.asarray(sweep.rt_ms, dtype=float),
        "final_x": np.asarray(sweep.final_x, dtype=float),
        "time_ms": np.asarray(sweep.time_ms, dtype=float),
        "metadata_json": np.asarray(json.dumps(_json_ready_metadata(dict(sweep.metadata)))),
    }
    if sweep.x_traj is not None:
        payload["x_traj"] = np.asarray(sweep.x_traj)
    if sweep.r_e is not None:
        payload["r_e"] = np.asarray(sweep.r_e, dtype=np.float32)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_simulation_sweep_npz(path: str | Path) -> AccumulatorSimulationSweep:
    """Load a coherence sweep from a compressed NumPy archive."""

    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        x_traj = np.asarray(data["x_traj"]) if "x_traj" in data.files else None
        r_e = np.asarray(data["r_e"], dtype=np.float32) if "r_e" in data.files else None
        return AccumulatorSimulationSweep(
            coherence_values=np.asarray(data["coherence_values"], dtype=float),
            choice=np.asarray(data["choice"]),
            hit_boundary=np.asarray(data["hit_boundary"]),
            rt_ms=np.asarray(data["rt_ms"], dtype=float),
            final_x=np.asarray(data["final_x"], dtype=float),
            time_ms=np.asarray(data["time_ms"], dtype=float),
            x_traj=x_traj,
            r_e=r_e,
            metadata=metadata,
        )


def _validate_common_task_args(
    *,
    dt_ddm: float,
    dt_model: float,
    dur: int,
    max_time: int | None,
    t_start: int,
    num_trials: int,
) -> tuple[int, int, int]:
    if int(num_trials) <= 0:
        raise ValueError("num_trials must be positive")
    if int(t_start) < 0:
        raise ValueError("t_start must be non-negative")
    if int(dur) <= 0:
        raise ValueError("dur must be positive")
    max_time = int(dur) if max_time is None else int(max_time)
    if max_time < int(dur):
        raise ValueError("max_time must be greater than or equal to dur")
    if int(t_start) > int(dur):
        raise ValueError("t_start must satisfy 0 <= t_start <= dur")

    dt_ddm_ms = decision_step_ms(float(dt_ddm))
    dt_model_ms = decision_step_ms(float(dt_model))
    if int(t_start) % dt_model_ms != 0:
        raise ValueError("t_start must align to the DT_MODEL sampling grid")
    if int(dur) % dt_model_ms != 0:
        raise ValueError("dur must align to the DT_MODEL sampling grid")
    if int(max_time) % dt_model_ms != 0:
        raise ValueError("max_time must align to the DT_MODEL sampling grid")
    return dt_ddm_ms, dt_model_ms, max_time


def _classify_circuit_choice(x_trace: np.ndarray, hit_trace: np.ndarray, *, boundary: float) -> int:
    hit_indices = np.flatnonzero(hit_trace)
    if hit_indices.size == 0:
        return 0
    hit_x = float(x_trace[int(hit_indices[0])])
    if hit_x >= float(boundary) - CHOICE_TOL:
        return 1
    if hit_x <= CHOICE_TOL:
        return -1
    return 1 if hit_x >= 0.5 * float(boundary) else -1


def _classify_hit_value(hit_x: float, *, boundary: float) -> int:
    if float(hit_x) >= float(boundary) - CHOICE_TOL:
        return 1
    if float(hit_x) <= CHOICE_TOL:
        return -1
    return 1 if float(hit_x) >= 0.5 * float(boundary) else -1


def _classify_terminal_value(final_x: float, *, boundary: float) -> int:
    midpoint = 0.5 * float(boundary)
    if float(final_x) > midpoint + CHOICE_TOL:
        return 1
    if float(final_x) < midpoint - CHOICE_TOL:
        return -1
    return 0


def _trial_rt_ms(hit_trace: np.ndarray, *, dt_model_ms: int, t_start: int) -> float:
    hit_indices = np.flatnonzero(hit_trace)
    if hit_indices.size == 0:
        return float("nan")
    return float(int(hit_indices[0]) * int(dt_model_ms) - int(t_start))


def _calibration_metadata(calibration: dict[str, Any] | None) -> dict[str, Any]:
    if calibration is None:
        return {}
    return {
        "kappa": float(calibration["kappa"]),
        "certificate_passed": bool(calibration["certificate_passed"]),
        "c_be_theta_max": float(calibration["c_be_theta_max"]),
    }


def _select_right_ramping_edge_indices(
    model: Any,
    *,
    x_start: float = R_E_SELECTION_START_X_DEFAULT,
    x_end: float = R_E_SELECTION_END_X_DEFAULT,
    r_start_max: float = R_E_SELECTION_START_MAX_DEFAULT,
    r_end_min: float = R_E_SELECTION_END_MIN_DEFAULT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return canonical right-ramping edge indices for compact FR saving."""

    start_pos = float(model.evidence_to_pos(float(x_start), model.gamma_E))
    end_pos = float(model.evidence_to_pos(float(x_end), model.gamma_E))
    start_profile = np.asarray(model.get_edge_stimulus_by_pos(start_pos, model.edge_type), dtype=float)
    end_profile = np.asarray(model.get_edge_stimulus_by_pos(end_pos, model.edge_type), dtype=float)
    selected = np.flatnonzero(
        (start_profile <= float(r_start_max)) & (end_profile >= float(r_end_min))
    ).astype(int)
    if selected.size == 0:
        raise ValueError(
            "Right-ramping edge-neuron selection produced an empty subset. "
            "Relax the FR thresholds or change the canonical x window."
        )
    metadata = {
        "selected_edge_indices": selected.tolist(),
        "selection_x_start": float(x_start),
        "selection_x_end": float(x_end),
        "selection_r_start_max": float(r_start_max),
        "selection_r_end_min": float(r_end_min),
        "selection_start_pos": start_pos,
        "selection_end_pos": end_pos,
        "num_selected_edge_units": int(selected.size),
    }
    return selected, metadata


def _build_circuit_baseline_params(
    *,
    coherence: float,
    drift_gain: float,
    noise_scale: float,
    dt_ddm: float,
    t_start: int,
    dur: int,
    max_time: int | None,
    seed: int,
    kappa: float | None = None,
    theta_margin: float = THETA_MARGIN_DEFAULT,
    x0: float = X0_DEFAULT,
    boundary: float = BOUNDARY_DEFAULT,
    mar: float = COMMIT_MARGIN_DEFAULT,
    c_eb: float | None = None,
) -> dict:
    from .default_params import build_stable_default_params

    params = build_stable_default_params()
    params["bump_pop"]["noise_scale_bump"] = 0.0
    params["edge_pop"]["noise_scale_edge"] = 0.0
    if c_eb is not None:
        params["edge_pop"]["c_EB"] = float(c_eb)

    c_be_params = {
        "mode": "target_diffusion",
        "theta_margin": float(theta_margin),
    }
    if kappa is not None:
        c_be_params["kappa"] = float(kappa)
    params["bump_pop"]["c_BE_params"] = c_be_params

    params["decision_space_params"]["decision_mode"] = "continuous"
    params["decision_space_params"]["decision_paradigm"] = "free_response"
    params["decision_space_params"]["t_start"] = int(t_start)
    params["decision_space_params"]["dur"] = int(dur)
    params["decision_space_params"]["max_time"] = int(dur) if max_time is None else int(max_time)
    params["decision_space_params"]["dt_DDM"] = float(dt_ddm)
    params["decision_space_params"]["x0"] = float(x0)
    params["decision_space_params"]["boundary"] = float(boundary)
    params["decision_space_params"]["mar"] = float(mar)
    params["decision_space_params"]["seed"] = int(seed)
    params["decision_space_params"]["drift_rate"] = float(drift_gain) * float(coherence)
    params["decision_space_params"]["noise_scale"] = float(noise_scale)
    return params


def prepare_circuit_target_diffusion_calibration(
    *,
    dt_ddm: float,
    t_start: int,
    dur: int,
    max_time: int | None = None,
    seed: int,
    theta_margin: float = THETA_MARGIN_DEFAULT,
    c_be_sweep: np.ndarray | None = None,
    min_accumulation_samples: int = MIN_ACCUMULATION_SAMPLES_DEFAULT,
    c_eb: float | None = None,
    mar: float = COMMIT_MARGIN_DEFAULT,
) -> dict:
    from CANN_DDM_model_rate_based import CANN_DDM_model

    c_be_sweep = (
        np.asarray(C_BE_SWEEP_DEFAULT, dtype=float)
        if c_be_sweep is None
        else np.asarray(c_be_sweep, dtype=float)
    )
    model = CANN_DDM_model(
        CANN_params=_build_circuit_baseline_params(
            coherence=0.0,
            drift_gain=0.0,
            noise_scale=0.0,
            dt_ddm=dt_ddm,
            t_start=t_start,
            dur=dur,
            max_time=max_time,
            seed=seed,
            kappa=None,
            theta_margin=theta_margin,
            mar=mar,
            c_eb=c_eb,
        )
    )
    return model.prepare_target_diffusion_mode(
        c_be_sweep=c_be_sweep,
        min_accumulation_samples=int(min_accumulation_samples),
    )


def simulate_circuit_trials(
    *,
    coherence: float,
    drift_gain: float,
    noise_scale: float,
    dt_ddm: float,
    dt_model: float,
    t_start: int,
    dur: int,
    max_time: int | None = None,
    num_trials: int,
    seed: int,
    decision_paradigm: str = "free_response",
    save_traj: bool = False,
    save_r_e: bool = False,
    calibration: dict[str, Any] | None = None,
    theta_margin: float = THETA_MARGIN_DEFAULT,
    c_be_sweep: np.ndarray | None = None,
    min_accumulation_samples: int = MIN_ACCUMULATION_SAMPLES_DEFAULT,
    x0: float = X0_DEFAULT,
    boundary: float = BOUNDARY_DEFAULT,
    mar: float = COMMIT_MARGIN_DEFAULT,
    c_eb: float | None = None,
    chunk_ms: int = CIRCUIT_CHUNK_MS_DEFAULT,
    progress_every: int | None = None,
) -> AccumulatorSimulationResult:
    from CANN_DDM_model_rate_based import CANN_DDM_model

    _, dt_model_ms, max_time = _validate_common_task_args(
        dt_ddm=dt_ddm,
        dt_model=dt_model,
        dur=dur,
        max_time=max_time,
        t_start=t_start,
        num_trials=num_trials,
    )
    decision_paradigm = str(decision_paradigm)
    if decision_paradigm not in {"free_response", "interrogation"}:
        raise ValueError(
            f"Unsupported decision_paradigm '{decision_paradigm}'. "
            "Supported paradigms: free_response, interrogation."
        )
    mar = float(mar)
    if not (0.0 <= mar < 0.5):
        raise ValueError("mar must satisfy 0 <= mar < 0.5")
    c_be_sweep = (
        np.asarray(C_BE_SWEEP_DEFAULT, dtype=float)
        if c_be_sweep is None
        else np.asarray(c_be_sweep, dtype=float)
    )
    chunk_ms = decision_step_ms(chunk_ms)
    if chunk_ms % dt_model_ms != 0:
        raise ValueError("chunk_ms must align to the DT_MODEL sampling grid")

    drift_rate = float(drift_gain) * float(coherence)
    calibration_provided = calibration is not None
    if calibration is None:
        calibration = prepare_circuit_target_diffusion_calibration(
            dt_ddm=dt_ddm,
            t_start=t_start,
            dur=dur,
            max_time=max_time,
            seed=seed,
            theta_margin=theta_margin,
            c_be_sweep=c_be_sweep,
            min_accumulation_samples=min_accumulation_samples,
            c_eb=c_eb,
            mar=mar,
        )
    kappa = float(calibration["kappa"])

    seed_sequence = np.random.SeedSequence(int(seed))
    child_sequences = seed_sequence.spawn(int(num_trials))
    progress_every = None if progress_every is None else int(progress_every)

    choice = np.zeros(int(num_trials), dtype=np.int8)
    hit_boundary = np.zeros(int(num_trials), dtype=bool)
    rt_ms = np.full(int(num_trials), np.nan, dtype=float)
    final_x = np.zeros(int(num_trials), dtype=float)
    total_steps = int(int(max_time) // dt_model_ms)
    time_ms = np.arange(total_steps, dtype=float) * float(dt_model_ms)
    x_traj = np.zeros((int(num_trials), total_steps), dtype=float) if save_traj else None
    r_e = None
    selected_edge_indices = None
    r_e_selection_metadata: dict[str, Any] = {}

    for idx, child in enumerate(child_sequences):
        if progress_every is not None and progress_every > 0 and idx % progress_every == 0:
            print(
                f"[simulate_circuit_trials] coherence={float(coherence):g} trial={idx}/{int(num_trials)}",
                file=sys.stderr,
                flush=True,
            )
        trial_seed = int(child.generate_state(1)[0])
        model = CANN_DDM_model(
            CANN_params=_build_circuit_baseline_params(
                coherence=coherence,
                drift_gain=drift_gain,
                noise_scale=noise_scale,
                dt_ddm=dt_ddm,
                t_start=t_start,
                dur=dur,
                max_time=max_time,
                seed=trial_seed,
                kappa=kappa,
                theta_margin=theta_margin,
                x0=x0,
                boundary=boundary,
                mar=mar,
                c_eb=c_eb,
            )
        )
        if save_r_e and selected_edge_indices is None:
            selected_edge_indices, r_e_selection_metadata = _select_right_ramping_edge_indices(model)
        if save_r_e and r_e is None:
            if selected_edge_indices is None:
                raise RuntimeError("Expected edge-neuron subset to be initialized before allocating r_e")
            r_e = np.zeros(
                (int(num_trials), total_steps, int(selected_edge_indices.size)),
                dtype=np.float32,
            )
        runner = model.build_runner(
            mon_vars=["x_E", "hit_boundary"] + (["r_E"] if save_r_e else []),
            progress_bar=False,
            dt=float(dt_model),
        )

        trial_trace = np.zeros(total_steps, dtype=float) if save_traj else None
        simulated_steps = 0
        trial_hit = False
        trial_choice = 0
        trial_rt = float("nan")
        trial_final_x = float("nan")

        while simulated_steps < total_steps:
            remaining_ms = int(max_time) - simulated_steps * dt_model_ms
            run_ms = min(chunk_ms, remaining_ms)
            runner.run(run_ms)

            x_chunk = np.asarray(runner.mon.x_E).reshape(-1)
            hit_chunk = np.asarray(runner.mon.hit_boundary).reshape(-1).astype(bool)
            r_e_chunk = np.asarray(runner.mon.r_E, dtype=np.float32) if save_r_e else None
            chunk_steps = int(x_chunk.size)
            start = simulated_steps
            stop = min(total_steps, start + chunk_steps)

            if save_traj and trial_trace is not None:
                trial_trace[start:stop] = x_chunk[: stop - start]
            if save_r_e and r_e is not None and r_e_chunk is not None:
                if selected_edge_indices is None:
                    raise RuntimeError("Expected edge-neuron subset to be initialized before saving r_e")
                r_e[idx, start:stop] = r_e_chunk[: stop - start, selected_edge_indices]

            if decision_paradigm == "free_response":
                hit_indices = np.flatnonzero(hit_chunk)
                if hit_indices.size > 0:
                    local_hit = int(hit_indices[0])
                    global_hit = start + local_hit
                    trial_hit = True
                    hit_x = float(x_chunk[local_hit])
                    trial_choice = _classify_hit_value(hit_x, boundary=boundary)
                    trial_rt = float(global_hit * dt_model_ms - int(t_start))
                    absorbing_x = float(boundary) if trial_choice == 1 else 0.0
                    trial_final_x = absorbing_x
                    if save_traj and trial_trace is not None:
                        trial_trace[global_hit:] = absorbing_x
                    break
            else:
                hit_indices = np.flatnonzero(hit_chunk)
                if hit_indices.size > 0:
                    trial_hit = True

            simulated_steps = stop
            if simulated_steps >= total_steps:
                trial_final_x = float(x_chunk[min(chunk_steps, total_steps - start) - 1])

        if not trial_hit and np.isnan(trial_final_x):
            trial_final_x = float(model.x_E[0])
        if decision_paradigm == "interrogation":
            trial_choice = _classify_terminal_value(trial_final_x, boundary=boundary)
            trial_rt = float("nan")

        choice[idx] = int(trial_choice)
        hit_boundary[idx] = bool(trial_hit)
        rt_ms[idx] = float(trial_rt)
        final_x[idx] = float(trial_final_x)
        if save_traj and x_traj is not None and trial_trace is not None:
            x_traj[idx] = trial_trace

    if progress_every is not None and progress_every > 0:
        print(
            f"[simulate_circuit_trials] coherence={float(coherence):g} trial={int(num_trials)}/{int(num_trials)} done",
            file=sys.stderr,
            flush=True,
        )

    metadata = {
        "model_type": "circuit",
        "coherence": float(coherence),
        "decision_paradigm": decision_paradigm,
        "drift_gain": float(drift_gain),
        "drift_rate": float(drift_rate),
        "noise_scale": float(noise_scale),
        "dt_ddm": float(dt_ddm),
        "dt_model": float(dt_model),
        "t_start": int(t_start),
        "dur": int(dur),
        "max_time": int(max_time),
        "num_trials": int(num_trials),
        "seed": int(seed),
        "boundary": float(boundary),
        "mar": float(mar),
        "c_eb": None if c_eb is None else float(c_eb),
        "x0": float(x0),
        "trajectory_source": "x_E",
        "theta_margin": float(theta_margin),
        "c_be_sweep": c_be_sweep.tolist(),
        "min_accumulation_samples": int(min_accumulation_samples),
        "chunk_ms": int(chunk_ms),
        "calibration_reused": bool(calibration_provided),
        "save_r_e": bool(save_r_e),
    }
    metadata.update(_calibration_metadata(calibration))
    if save_r_e:
        metadata.update(r_e_selection_metadata)
    result = AccumulatorSimulationResult(
        choice=choice,
        hit_boundary=hit_boundary,
        rt_ms=rt_ms,
        final_x=final_x,
        time_ms=np.asarray(time_ms, dtype=float),
        x_traj=x_traj,
        r_e=r_e,
        metadata=metadata,
    )
    return result


def simulate_circuit_condition(**kwargs) -> AccumulatorSimulationResult:
    """Compatibility alias for the older one-condition circuit entrypoint."""

    return simulate_circuit_trials(**kwargs)


def _simulate_continuous_ddm_trials(
    *,
    config: ContinuousDDMConfig,
    dt_DDM: float,
    dur: int,
    max_time: int | None,
    t_start: int,
    x0: float,
    boundary: float,
    num_trials: int,
    decision_paradigm: str,
    seed: int | None = None,
    return_traj: bool = False,
) -> AccumulatorSimulationResult:
    """Simulate a batch of continuous DDM trials under the requested paradigm."""

    total_time = int(dur) if max_time is None else int(max_time)
    if total_time <= 0:
        raise ValueError("max_time must be positive")
    if int(num_trials) <= 0:
        raise ValueError("num_trials must be positive")
    if int(t_start) < 0 or int(t_start) > int(dur):
        raise ValueError("t_start must satisfy 0 <= t_start <= dur")
    if int(dur) > total_time:
        raise ValueError("max_time must be greater than or equal to dur")
    if float(boundary) <= 0.0:
        raise ValueError("boundary must be positive")
    if not (0.0 <= float(x0) <= float(boundary)):
        raise ValueError("x0 must lie within [0, boundary]")
    if float(config.noise_scale) < 0.0:
        raise ValueError("noise_scale must be non-negative")
    if decision_paradigm not in {"free_response", "interrogation"}:
        raise ValueError(
            f"Unsupported decision_paradigm '{decision_paradigm}'. "
            "Supported paradigms: free_response, interrogation."
        )

    step_ms = decision_step_ms(dt_DDM)
    dt_s = float(dt_DDM) * 1e-3
    num_trials = int(num_trials)
    t_start = int(t_start)
    dur = int(dur)
    boundary = float(boundary)
    x0 = float(x0)

    rng = np.random.default_rng(seed)
    x_curr = np.full(num_trials, x0, dtype=float)
    choice = np.zeros(num_trials, dtype=np.int8)
    hit_boundary = np.zeros(num_trials, dtype=bool)
    rt_ms = np.full(num_trials, np.nan, dtype=float)
    time_ms = np.arange(total_time, dtype=float)
    x_traj = np.empty((num_trials, total_time), dtype=float) if return_traj else None

    step_starts = np.arange(0, total_time, step_ms, dtype=int)
    for start in step_starts:
        stop = min(start + step_ms, total_time)
        if start < t_start:
            if x_traj is not None:
                x_traj[:, start:stop] = x_curr[:, None]
            continue

        drift_increment = float(config.drift_rate) * dt_s if start < dur else 0.0
        noise_increment = float(config.noise_scale) * np.sqrt(dt_s) * rng.standard_normal(num_trials)
        increments = drift_increment + noise_increment

        if decision_paradigm == "free_response":
            active_trials = ~hit_boundary
            if np.any(active_trials):
                x_curr[active_trials] += increments[active_trials]

                right_hits = active_trials & (x_curr >= boundary)
                left_hits = active_trials & (x_curr <= 0.0)
                newly_hit = right_hits | left_hits

                x_curr[right_hits] = boundary
                x_curr[left_hits] = 0.0
                choice[right_hits] = 1
                choice[left_hits] = -1
                hit_boundary[newly_hit] = True
                rt_ms[newly_hit] = float(start - t_start)
        else:
            x_curr += increments
            hit_boundary |= (x_curr >= boundary) | (x_curr <= 0.0)

        if x_traj is not None:
            x_traj[:, start:stop] = x_curr[:, None]

    if decision_paradigm == "interrogation":
        choice = np.asarray([_classify_terminal_value(value, boundary=boundary) for value in x_curr], dtype=np.int8)

    metadata = {
        "model_type": "ddm",
        "decision_mode": "continuous",
        "decision_paradigm": decision_paradigm,
        "drift_rate": float(config.drift_rate),
        "noise_scale": float(config.noise_scale),
        "dt_ddm": float(dt_DDM),
        "dt_model": 1.0,
        "t_start": int(t_start),
        "dur": int(dur),
        "max_time": int(total_time),
        "num_trials": int(num_trials),
        "seed": None if seed is None else int(seed),
        "boundary": float(boundary),
        "x0": float(x0),
        "trajectory_source": "x",
    }
    return AccumulatorSimulationResult(
        choice=choice,
        hit_boundary=hit_boundary,
        rt_ms=rt_ms,
        final_x=x_curr.copy(),
        time_ms=time_ms,
        x_traj=x_traj,
        r_e=None,
        metadata=metadata,
    )


def _simulate_pulse_ddm_trials(
    *,
    config: PulseDDMConfig,
    dt_DDM: float,
    dur: int,
    max_time: int | None,
    t_start: int,
    x0: float,
    boundary: float,
    num_trials: int,
    decision_paradigm: str,
    seed: int | None = None,
    return_traj: bool = False,
) -> AccumulatorSimulationResult:
    """Simulate a batch of pulse-based DDM trials under the requested paradigm."""

    total_time = int(dur) if max_time is None else int(max_time)
    if total_time <= 0:
        raise ValueError("max_time must be positive")
    if int(num_trials) <= 0:
        raise ValueError("num_trials must be positive")
    if int(t_start) < 0 or int(t_start) > int(dur):
        raise ValueError("t_start must satisfy 0 <= t_start <= dur")
    if int(dur) > total_time:
        raise ValueError("max_time must be greater than or equal to dur")
    if float(boundary) <= 0.0:
        raise ValueError("boundary must be positive")
    if not (0.0 <= float(x0) <= float(boundary)):
        raise ValueError("x0 must lie within [0, boundary]")
    if float(config.noise_scale) < 0.0:
        raise ValueError("noise_scale must be non-negative")
    if float(config.lambda_commit) < 0.0:
        raise ValueError("lambda_commit must be non-negative")
    if not (0.0 <= float(config.lambda_click_L) <= 1.0):
        raise ValueError("lambda_click_L must lie in [0, 1] for the 1 ms Bernoulli approximation")
    if not (0.0 <= float(config.lambda_click_R) <= 1.0):
        raise ValueError("lambda_click_R must lie in [0, 1] for the 1 ms Bernoulli approximation")
    if float(config.delta_click_x) < 0.0:
        raise ValueError("delta_click_x must be non-negative")
    if decision_paradigm not in {"free_response", "interrogation"}:
        raise ValueError(
            f"Unsupported decision_paradigm '{decision_paradigm}'. "
            "Supported paradigms: free_response, interrogation."
        )

    step_ms = decision_step_ms(dt_DDM)
    dt_s = float(dt_DDM) * 1e-3
    num_trials = int(num_trials)
    t_start = int(t_start)
    dur = int(dur)
    boundary = float(boundary)
    x0 = float(x0)
    center_x = 0.5 * boundary

    rng = np.random.default_rng(seed)
    x_curr = np.full(num_trials, x0, dtype=float)
    choice = np.zeros(num_trials, dtype=np.int8)
    hit_boundary = np.zeros(num_trials, dtype=bool)
    rt_ms = np.full(num_trials, np.nan, dtype=float)
    time_ms = np.arange(total_time, dtype=float)
    x_traj = np.empty((num_trials, total_time), dtype=float) if return_traj else None
    click_count_L = np.zeros(num_trials, dtype=int)
    click_count_R = np.zeros(num_trials, dtype=int)

    noise_step_mask = np.zeros(total_time, dtype=bool)
    step_starts = np.arange(0, total_time, step_ms, dtype=int)
    noise_step_mask[step_starts[step_starts >= t_start]] = True

    for t in range(total_time):
        if t < t_start:
            if x_traj is not None:
                x_traj[:, t] = x_curr
            continue

        click_active = t < dur
        if click_active:
            click_R = rng.binomial(1, float(config.lambda_click_R), num_trials)
            click_L = rng.binomial(1, float(config.lambda_click_L), num_trials)
        else:
            click_R = np.zeros(num_trials, dtype=int)
            click_L = np.zeros(num_trials, dtype=int)
        click_count_R += click_R
        click_count_L += click_L

        commit_increments = float(config.lambda_commit) * (x_curr - center_x) * 1e-3
        click_increments = float(config.delta_click_x) * (click_R - click_L)
        increments = click_increments + commit_increments
        if noise_step_mask[t]:
            increments = increments + float(config.noise_scale) * np.sqrt(dt_s) * rng.standard_normal(num_trials)

        if decision_paradigm == "free_response":
            active_trials = ~hit_boundary
            if np.any(active_trials):
                x_curr[active_trials] += increments[active_trials]

                right_hits = active_trials & (x_curr >= boundary)
                left_hits = active_trials & (x_curr <= 0.0)
                newly_hit = right_hits | left_hits

                x_curr[right_hits] = boundary
                x_curr[left_hits] = 0.0
                choice[right_hits] = 1
                choice[left_hits] = -1
                hit_boundary[newly_hit] = True
                rt_ms[newly_hit] = float(t - t_start)
        else:
            x_curr += increments
            hit_boundary |= (x_curr >= boundary) | (x_curr <= 0.0)

        if x_traj is not None:
            x_traj[:, t] = x_curr

    if decision_paradigm == "interrogation":
        choice = np.asarray([_classify_terminal_value(value, boundary=boundary) for value in x_curr], dtype=np.int8)

    metadata = {
        "model_type": "ddm",
        "decision_mode": "discrete",
        "decision_paradigm": decision_paradigm,
        "lambda_click_L": float(config.lambda_click_L),
        "lambda_click_R": float(config.lambda_click_R),
        "delta_click_x": float(config.delta_click_x),
        "noise_scale": float(config.noise_scale),
        "lambda_commit": float(config.lambda_commit),
        "dt_ddm": float(dt_DDM),
        "dt_model": 1.0,
        "t_start": int(t_start),
        "dur": int(dur),
        "max_time": int(total_time),
        "num_trials": int(num_trials),
        "seed": None if seed is None else int(seed),
        "boundary": float(boundary),
        "x0": float(x0),
        "trajectory_source": "x",
        "click_time_step_ms": 1.0,
        "click_count_window": "full_stimulus",
        "click_count_L": click_count_L.copy(),
        "click_count_R": click_count_R.copy(),
        "net_click_count": (click_count_R - click_count_L).copy(),
    }
    return AccumulatorSimulationResult(
        choice=choice,
        hit_boundary=hit_boundary,
        rt_ms=rt_ms,
        final_x=x_curr.copy(),
        time_ms=time_ms,
        x_traj=x_traj,
        r_e=None,
        metadata=metadata,
    )


def simulate_ddm_trials(
    *,
    decision_mode: str,
    config: ContinuousDDMConfig | PulseDDMConfig,
    dt_DDM: float,
    dur: int,
    max_time: int | None = None,
    t_start: int,
    x0: float,
    boundary: float,
    num_trials: int,
    decision_paradigm: str = "free_response",
    seed: int | None = None,
    return_traj: bool = False,
) -> AccumulatorSimulationResult:
    """Dispatch task-level DDM simulation by decision mode."""

    mode = str(decision_mode)
    if mode == "continuous":
        if not isinstance(config, ContinuousDDMConfig):
            raise TypeError("continuous mode requires config=ContinuousDDMConfig(...)")
        return _simulate_continuous_ddm_trials(
            config=config,
            dt_DDM=dt_DDM,
            dur=dur,
            max_time=max_time,
            t_start=t_start,
            x0=x0,
            boundary=boundary,
            num_trials=num_trials,
            decision_paradigm=decision_paradigm,
            seed=seed,
            return_traj=return_traj,
        )
    if mode == "discrete":
        if not isinstance(config, PulseDDMConfig):
            raise TypeError("discrete mode requires config=PulseDDMConfig(...)")
        return _simulate_pulse_ddm_trials(
            config=config,
            dt_DDM=dt_DDM,
            dur=dur,
            max_time=max_time,
            t_start=t_start,
            x0=x0,
            boundary=boundary,
            num_trials=num_trials,
            decision_paradigm=decision_paradigm,
            seed=seed,
            return_traj=return_traj,
        )
    raise ValueError(f"Unsupported decision_mode '{decision_mode}'. Supported modes: continuous, discrete.")


def simulate_absorbed_ddm_trials(
    *,
    drift_rate: float,
    noise_scale: float,
    dt_DDM: float,
    dur: int,
    t_start: int,
    x0: float,
    boundary: float,
    num_trials: int,
    seed: int | None = None,
    return_traj: bool = False,
) -> AccumulatorSimulationResult:
    """Compatibility wrapper for the legacy continuous absorbed-DDM entrypoint."""

    return simulate_ddm_trials(
        decision_mode="continuous",
        config=ContinuousDDMConfig(
            drift_rate=float(drift_rate),
            noise_scale=float(noise_scale),
        ),
        dt_DDM=dt_DDM,
        dur=dur,
        t_start=t_start,
        x0=x0,
        boundary=boundary,
        num_trials=num_trials,
        seed=seed,
        return_traj=return_traj,
    )

 
