#!/usr/bin/env python3
"""Orchestrate the SCC psychometric workflow from one Python entrypoint."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.combine_psychometric_model_datasets import main as combine_model_datasets_main
from scripts.generate_cddm_psychometric_dataset import main as generate_cddm_dataset_main
from scripts.merge_circuit_psychometric_outputs import main as merge_circuit_outputs_main
from scripts.simulate_psychometric_data_cDDM import main as simulate_cddm_main


DEFAULT_COHERENCE_VALUES = "-1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0"
DEFAULT_PYTHON = "/projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python"


@dataclass(frozen=True)
class WorkflowPaths:
    run_root: Path
    ddm_root: Path
    circuit_root: Path
    jobs_root: Path
    logs_root: Path
    manifest_path: Path
    finalizer_logs_root: Path
    finalizer_metadata_path: Path
    worker_job_script: Path
    finalizer_job_script: Path


def repo_root_from_env() -> Path:
    env_repo_root = os.environ.get("REPO_ROOT")
    if env_repo_root:
        return Path(env_repo_root).resolve()
    return REPO_ROOT


def default_python_bin() -> str:
    return os.environ.get("CANN_DDM_PYTHON", DEFAULT_PYTHON)


def parse_coherence_values(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("coherence-values must contain at least one numeric value")
    return values


def coherence_slug(value: float) -> str:
    sign = "p" if value >= 0.0 else "m"
    magnitude = f"{abs(float(value)):.6f}".rstrip("0").rstrip(".") or "0"
    return f"{sign}{magnitude.replace('.', 'p')}"


def resolve_paths(*, repo_root: Path, output_root: str, run_name: str) -> WorkflowPaths:
    run_base = Path(output_root)
    if not run_base.is_absolute():
        run_base = repo_root / run_base
    run_root = (run_base / run_name).resolve()
    ddm_root = run_root / "ddm"
    circuit_root = run_root / "circuit"
    return WorkflowPaths(
        run_root=run_root,
        ddm_root=ddm_root,
        circuit_root=circuit_root,
        jobs_root=circuit_root / "jobs",
        logs_root=circuit_root / "logs",
        manifest_path=circuit_root / "submission_manifest.tsv",
        finalizer_logs_root=run_root / "finalizer_logs",
        finalizer_metadata_path=run_root / "finalizer_metadata.env",
        worker_job_script=run_root / "run_psychometric_worker_job.sh",
        finalizer_job_script=run_root / "run_psychometric_finalizer_job.sh",
    )


def ensure_submit_valid(args: argparse.Namespace) -> None:
    if int(args.num_trials) <= 0:
        raise ValueError("num-trials must be positive")
    if int(args.num_batches) <= 0:
        raise ValueError("num-batches must be positive")
    if int(args.circuit_batch_trials) <= 0:
        raise ValueError("circuit-batch-trials must be positive")
    if int(args.num_batches) * int(args.circuit_batch_trials) != int(args.num_trials):
        raise ValueError(
            "Expected num-batches * circuit-batch-trials == num-trials, "
            f"got {args.num_batches} * {args.circuit_batch_trials} != {args.num_trials}"
        )


def write_metadata(paths: WorkflowPaths, *, initial: bool, **extra: str) -> None:
    lines = []
    if initial:
        lines.extend(
            [
                f"run_root={paths.run_root}",
                f"ddm_root={paths.ddm_root}",
                f"circuit_root={paths.circuit_root}",
                f"finalizer_logs_root={paths.finalizer_logs_root}",
                f"submission_time={datetime.now().astimezone().isoformat(timespec='seconds')}",
            ]
        )
    for key, value in extra.items():
        lines.append(f"{key}={value}")
    payload = "\n".join(lines) + "\n"
    if initial:
        paths.finalizer_metadata_path.write_text(payload, encoding="ascii")
        return
    with paths.finalizer_metadata_path.open("a", encoding="ascii") as handle:
        handle.write(payload)


def write_wrapper_script(path: Path, *, repo_root: Path, python_bin: str, subcommand: str) -> None:
    content = "\n".join(
        [
            "#!/bin/bash -l",
            "set -euo pipefail",
            f"export REPO_ROOT={shlex.quote(str(repo_root))}",
            f"export CANN_DDM_PYTHON={shlex.quote(str(python_bin))}",
            f"cd {shlex.quote(str(repo_root))}",
            (
                f"exec {shlex.quote(str(python_bin))} "
                f"{shlex.quote(str(repo_root / 'scripts' / 'run_psychometric_workflow.py'))} "
                f"{shlex.quote(subcommand)} \"$@\""
            ),
            "",
        ]
    )
    path.write_text(content, encoding="ascii")
    path.chmod(0o755)


def generate_ddm_bundle(args: argparse.Namespace, paths: WorkflowPaths) -> None:
    argv = [
        "--run-name",
        "ddm",
        "--output-root",
        str(paths.run_root),
        f"--coherence-values={args.coherence_values}",
        "--drift-gain",
        str(args.drift_gain),
        "--noise-scale",
        str(args.noise_scale),
        "--dt-ddm",
        str(args.dt_ddm),
        "--dt-model",
        str(args.dt_model),
        "--t-start",
        str(args.t_start),
        "--dur",
        str(args.dur),
        "--x0",
        str(args.x0),
        "--boundary",
        str(args.boundary),
        "--num-trials",
        str(args.num_trials),
        "--seed",
        str(args.seed),
    ]
    if args.save_traj:
        argv.append("--save-traj")
    generate_cddm_dataset_main(argv)


def worker_job_name(*, coherence: float, batch_index: int, batch_trials: int) -> str:
    return f"psweep_{coherence_slug(coherence)}_b{batch_index:03d}_n{batch_trials}"


def worker_output_dir(*, paths: WorkflowPaths, coherence: float, batch_index: int) -> Path:
    slug = coherence_slug(coherence)
    output_dir = paths.jobs_root / f"coh_{slug}" / f"batch_{batch_index:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def command_prepare(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_env()
    python_bin = str(Path(args.python_bin).resolve())
    args.circuit_batch_trials = (
        int(args.circuit_batch_trials) if args.circuit_batch_trials is not None else int(args.num_trials)
    )
    ensure_submit_valid(args)

    paths = resolve_paths(repo_root=repo_root, output_root=str(args.output_root), run_name=str(args.run_name))
    paths.run_root.mkdir(parents=True, exist_ok=True)
    paths.jobs_root.mkdir(parents=True, exist_ok=True)
    paths.logs_root.mkdir(parents=True, exist_ok=True)
    paths.finalizer_logs_root.mkdir(parents=True, exist_ok=True)

    paths.manifest_path.write_text(
        "coherence\tcoherence_slug\tbatch_index\tbatch_num_trials\tseed\tjob_name\tjob_id\toutput_dir\n",
        encoding="ascii",
    )
    write_metadata(paths, initial=True, finalizer_name=f"psweep_finalize_{args.run_name}")
    write_wrapper_script(paths.worker_job_script, repo_root=repo_root, python_bin=python_bin, subcommand="worker")
    write_wrapper_script(paths.finalizer_job_script, repo_root=repo_root, python_bin=python_bin, subcommand="finalize")

    print(f"Generating DDM dataset locally under {paths.ddm_root}")
    generate_ddm_bundle(args, paths)

    coherences = parse_coherence_values(str(args.coherence_values))
    coherence_seqs = np.random.SeedSequence(int(args.seed)).spawn(len(coherences))
    manifest_rows: list[str] = []

    for coherence_index, coherence in enumerate(coherences):
        batch_seqs = coherence_seqs[coherence_index].spawn(int(args.num_batches))
        for batch_index, batch_seq in enumerate(batch_seqs):
            batch_seed = int(batch_seq.generate_state(1)[0])
            output_dir = worker_output_dir(paths=paths, coherence=coherence, batch_index=batch_index)
            job_name = worker_job_name(
                coherence=coherence,
                batch_index=batch_index,
                batch_trials=int(args.circuit_batch_trials),
            )
            manifest_rows.append(
                "\t".join(
                    [
                        str(coherence),
                        coherence_slug(coherence),
                        str(batch_index),
                        str(int(args.circuit_batch_trials)),
                        str(batch_seed),
                        job_name,
                        "",
                        str(output_dir),
                    ]
                )
            )

    with paths.manifest_path.open("a", encoding="ascii") as handle:
        for row in manifest_rows:
            handle.write(row + "\n")

    expected_lines = 1 + len(coherences) * int(args.num_batches)
    actual_lines = sum(1 for _ in paths.manifest_path.open("r", encoding="ascii"))
    if actual_lines != expected_lines:
        raise RuntimeError(
            f"Manifest line count mismatch: expected {expected_lines}, found {actual_lines}"
        )

    print(f"Run root: {paths.run_root}")
    print(f"Worker job script: {paths.worker_job_script}")
    print(f"Finalizer job script: {paths.finalizer_job_script}")
    print(f"Manifest: {paths.manifest_path}")
    print(f"Finalizer metadata: {paths.finalizer_metadata_path}")
    print("Circuit worker shard outputs:")
    print("  $RUN_ROOT/circuit/jobs/coh_*/batch_*/conditions/*.npz")
    print("  $RUN_ROOT/circuit/jobs/coh_*/batch_*/summary.csv")
    print("  $RUN_ROOT/circuit/jobs/coh_*/batch_*/config.json")
    print("Merged circuit sweep outputs after the finalizer merge step:")
    print(f"  {paths.circuit_root / 'dataset.npz'}")
    print(f"  {paths.circuit_root / 'summary.csv'}")
    print(f"  {paths.circuit_root / 'config.json'}")
    print("Final combined outputs after successful merge + combine:")
    print(f"  {paths.run_root / 'dataset.npz'}")
    print(f"  {paths.run_root / 'summary.csv'}")
    print(f"  {paths.run_root / 'config.json'}")
    return 0


def command_record_submissions(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest_path).resolve()
    metadata_path = Path(args.metadata_path).resolve()
    rows = manifest_path.read_text(encoding="ascii").splitlines()
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    expected_count = len(rows) - 1
    job_ids = [item for item in str(args.worker_job_ids).split(",") if item]
    if expected_count != len(job_ids):
        raise ValueError(
            f"Expected {expected_count} worker job ids, found {len(job_ids)}"
        )

    updated_lines = [rows[0]]
    for line, job_id in zip(rows[1:], job_ids, strict=True):
        fields = line.split("\t")
        if len(fields) != 8:
            raise ValueError(f"Unexpected manifest row shape: {line}")
        fields[6] = job_id
        updated_lines.append("\t".join(fields))
    manifest_path.write_text("\n".join(updated_lines) + "\n", encoding="ascii")

    with metadata_path.open("a", encoding="ascii") as handle:
        handle.write(f"worker_job_ids={args.worker_job_ids}\n")
        handle.write(f"finalizer_job_id={args.finalizer_job_id}\n")
    return 0


def command_worker(args: argparse.Namespace) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    argv = [
        "--model",
        str(args.model),
        "--coherence-values",
        str(args.coherence),
        "--drift-gain",
        str(args.drift_gain),
        "--noise-scale",
        str(args.noise_scale),
        "--dt-ddm",
        str(args.dt_ddm),
        "--dt-model",
        str(args.dt_model),
        "--t-start",
        str(args.t_start),
        "--dur",
        str(args.dur),
        "--x0",
        str(args.x0),
        "--boundary",
        str(args.boundary),
        "--num-trials",
        str(args.num_trials),
        "--seed",
        str(args.seed),
        "--chunk-ms",
        str(args.chunk_ms),
        "--resume",
        "--output-dir",
        str(Path(args.output_dir).resolve()),
    ]
    if args.save_traj:
        argv.append("--save-traj")
    return simulate_cddm_main(argv)


def command_finalize(args: argparse.Namespace) -> int:
    repo_root = repo_root_from_env()
    merge_script = repo_root / "scripts" / "merge_circuit_psychometric_outputs.py"
    combine_script = repo_root / "scripts" / "combine_psychometric_model_datasets.py"

    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] Starting psychometric finalizer")
    print(f"repo_root={repo_root}")
    print(f"run_root={Path(args.run_root).resolve()}")
    print(f"ddm_root={Path(args.ddm_root).resolve()}")
    print(f"circuit_root={Path(args.circuit_root).resolve()}")
    print(f"merge_script={merge_script}")
    print(f"combine_script={combine_script}")

    if not merge_script.is_file():
        raise FileNotFoundError(f"Missing merge script: {merge_script}")
    if not combine_script.is_file():
        raise FileNotFoundError(f"Missing combine script: {combine_script}")
    if not Path(args.ddm_root).is_dir():
        raise FileNotFoundError(f"Missing DDM root: {args.ddm_root}")
    if not Path(args.circuit_root).is_dir():
        raise FileNotFoundError(f"Missing circuit root: {args.circuit_root}")

    merge_circuit_outputs_main([str(Path(args.circuit_root).resolve())])
    combine_model_datasets_main(
        [
            "--ddm-root",
            str(Path(args.ddm_root).resolve()),
            "--circuit-root",
            str(Path(args.circuit_root).resolve()),
            "--output-root",
            str(Path(args.run_root).resolve()),
            "--delete-intermediates",
        ]
    )

    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] Finished psychometric finalizer")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("run_name")
    prepare_parser.add_argument("--num-trials", type=int, default=200)
    prepare_parser.add_argument("--num-batches", type=int, default=1)
    prepare_parser.add_argument("--circuit-batch-trials", type=int)
    prepare_parser.add_argument("--coherence-values", default=DEFAULT_COHERENCE_VALUES)
    prepare_parser.add_argument("--drift-gain", type=float, default=1.0)
    prepare_parser.add_argument("--noise-scale", type=float, default=0.5)
    prepare_parser.add_argument("--dt-ddm", type=float, default=5.0)
    prepare_parser.add_argument("--dt-model", type=float, default=1.0)
    prepare_parser.add_argument("--t-start", type=int, default=10)
    prepare_parser.add_argument("--dur", type=int, default=2000)
    prepare_parser.add_argument("--x0", type=float, default=0.5)
    prepare_parser.add_argument("--boundary", type=float, default=1.0)
    prepare_parser.add_argument("--seed", type=int, default=201)
    prepare_parser.add_argument("--chunk-ms", type=int, default=1000)
    prepare_parser.add_argument("--output-root", default="results/psychometric")
    prepare_parser.add_argument("--python-bin", default=default_python_bin())
    prepare_parser.add_argument("--save-traj", action="store_true")
    prepare_parser.set_defaults(func=command_prepare)

    record_parser = subparsers.add_parser("record-submissions")
    record_parser.add_argument("--manifest-path", required=True)
    record_parser.add_argument("--metadata-path", required=True)
    record_parser.add_argument("--worker-job-ids", required=True)
    record_parser.add_argument("--finalizer-job-id", required=True)
    record_parser.set_defaults(func=command_record_submissions)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--model", choices=("ddm", "circuit"), default="circuit")
    worker_parser.add_argument("--coherence", type=float, required=True)
    worker_parser.add_argument("--num-trials", type=int, required=True)
    worker_parser.add_argument("--output-dir", required=True)
    worker_parser.add_argument("--drift-gain", type=float, required=True)
    worker_parser.add_argument("--noise-scale", type=float, required=True)
    worker_parser.add_argument("--dt-ddm", type=float, required=True)
    worker_parser.add_argument("--dt-model", type=float, default=1.0)
    worker_parser.add_argument("--t-start", type=int, required=True)
    worker_parser.add_argument("--dur", type=int, required=True)
    worker_parser.add_argument("--x0", type=float, default=0.5)
    worker_parser.add_argument("--boundary", type=float, default=1.0)
    worker_parser.add_argument("--seed", type=int, required=True)
    worker_parser.add_argument("--chunk-ms", type=int, default=1000)
    worker_parser.add_argument("--save-traj", action="store_true")
    worker_parser.set_defaults(func=command_worker)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-root", required=True)
    finalize_parser.add_argument("--ddm-root", required=True)
    finalize_parser.add_argument("--circuit-root", required=True)
    finalize_parser.set_defaults(func=command_finalize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
