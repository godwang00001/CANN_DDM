"""Backward-compatible wrappers for the shared accumulator simulation module."""

from __future__ import annotations

from .accumulator_simulation import (
    AccumulatorSimulationResult as DDMSimulationResult,
    decision_step_ms,
    load_simulation_result_npz,
    save_simulation_result_npz,
    simulate_absorbed_ddm_trials,
    simulate_ddm_trials,
)

__all__ = [
    "DDMSimulationResult",
    "decision_step_ms",
    "load_simulation_result_npz",
    "save_simulation_result_npz",
    "simulate_absorbed_ddm_trials",
    "simulate_ddm_trials",
]
