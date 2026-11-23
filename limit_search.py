"""Run one guarded DMRG or statevector-VQE limit test at a time.

Examples from the project root:
    python limit_search.py dmrg --L 120 --chi 16 --g 0.5
    python limit_search.py dmrg --L 40 --chi 96 --g 0.5
    python limit_search.py vqe --L 14 --g 0.5 --layers 2 --steps 100 --budget-seconds 120

The resource estimates are guardrails, not guarantees. Never launch an
ascending batch automatically; inspect each record before increasing size.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import quimb.tensor as qtn


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "tutorial_faithful" / "limit_search"
RESULTS.mkdir(parents=True, exist_ok=True)
PROCESS = psutil.Process()

# The free-memory reserve is a user-configurable safety margin, not a measured
# hardware limit. Model-size admission is based on the live resource estimate.
DEFAULT_FREE_RAM_RESERVE_GIB = 1.75
ABSOLUTE_MIN_FREE_RAM_RESERVE_GIB = 1.5


class ResourceMonitor:
    """Sample process RSS and system-available RAM during a model run."""

    def __init__(self, interval_seconds: float = 0.2):
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_rss_mib = 0.0
        self.minimum_available_gib = float("inf")
        self.shutdown_timed_out = False

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                self.peak_rss_mib = max(
                    self.peak_rss_mib, PROCESS.memory_info().rss / 2**20
                )
                self.minimum_available_gib = min(
                    self.minimum_available_gib,
                    psutil.virtual_memory().available / 2**30,
                )
            except psutil.Error:
                pass
            self.stop_event.wait(self.interval_seconds)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=max(2.0, 5.0 * self.interval_seconds))
        self.shutdown_timed_out = self.thread.is_alive()


def installed_version(distribution: str) -> str:
    """Return a package version without making missing metadata fatal."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not found"


def snapshot() -> dict[str, float]:
    return {
        "available_gib": psutil.virtual_memory().available / 2**30,
        "rss_mib": PROCESS.memory_info().rss / 2**20,
    }


def append_csv(filename: str, row: dict) -> None:
    path = RESULTS / filename
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        old_fields = reader.fieldnames or []
        old_rows = list(reader)
    fieldnames = old_fields + [field for field in row if field not in old_fields]
    old_rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(old_rows)


def base_record(method: str, L: int, g: float) -> dict:
    before = snapshot()
    return {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": method,
        "L": L,
        "g": g,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_identifier": platform.processor(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "installed_ram_gib": psutil.virtual_memory().total / 2**30,
        "numpy_version": np.__version__,
        "psutil_version": psutil.__version__,
        "quimb_version": installed_version("quimb"),
        "compute_backend": (
            "Quimb default CPU backend" if method == "DMRG/MPS"
            else "TensorCircuit with TensorFlow statevector backend"
        ),
        "available_gib_before": before["available_gib"],
        "rss_before_mib": before["rss_mib"],
        "status": "NOT_STARTED",
        "elapsed_seconds": 0.0,
        "rss_after_mib": "",
        "available_gib_after": "",
        "error": "",
    }


def finish_record(row: dict, started: float, status: str,
                  error: str = "", monitor=None) -> dict:
    if monitor is not None:
        monitor.stop()
    after = snapshot()
    row.update({
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "rss_after_mib": after["rss_mib"],
        "available_gib_after": after["available_gib"],
        "error": error,
    })
    if monitor is not None:
        row["rss_peak_sampled_mib"] = monitor.peak_rss_mib
        row["minimum_available_gib_sampled"] = monitor.minimum_available_gib
        row["resource_monitor_thread_alive_after_stop"] = monitor.shutdown_timed_out
    return row


def run_dmrg(args) -> dict:
    started = time.perf_counter()
    row = base_record("DMRG/MPS", args.L, args.g)
    log2_estimate = (
        math.log2(512.0) + math.log2(max(args.L, 1))
        + 2.0 * math.log2(max(args.chi, 1)) - 30.0
    )
    estimate_gib = math.inf if log2_estimate > 1023 else 2.0**log2_estimate
    row.update({
        "chi": args.chi,
        "max_sweeps": args.max_sweeps,
        "tolerance": args.tolerance,
        "workspace_estimate_gib_heuristic": estimate_gib,
    })

    if args.L < 2 or args.chi < 2 or args.max_sweeps < 1:
        row["status"] = "SKIPPED_INVALID_CONFIGURATION"
        append_csv("dmrg_limit_search.csv", row)
        return row
    reserve_gib = max(args.min_free_gib, ABSOLUTE_MIN_FREE_RAM_RESERVE_GIB)
    row["minimum_free_ram_reserve_gib"] = reserve_gib
    if row["available_gib_before"] - estimate_gib < reserve_gib:
        row["status"] = "SKIPPED_MEMORY_RESERVE"
        append_csv("dmrg_limit_search.csv", row)
        return row

    print(
        f"DMRG admitted: L={args.L}, chi={args.chi}, "
        f"estimate={estimate_gib:.3f} GiB; building Hamiltonian.",
        flush=True,
    )
    monitor = ResourceMonitor()
    monitor.peak_rss_mib = row["rss_before_mib"]
    monitor.minimum_available_gib = row["available_gib_before"]
    try:
        monitor.start()
        print("Resource monitor started; constructing MPO.", flush=True)
        H = qtn.MPO_ham_ising(args.L, j=1.0, bx=args.g)
        print("MPO constructed; initializing DMRG.", flush=True)
        solver = qtn.DMRG2(H, bond_dims=[args.chi])
        print("DMRG initialized; beginning sweeps.", flush=True)
        solver.solve(max_sweeps=args.max_sweeps, tol=args.tolerance, verbosity=0)
        print("DMRG sweeps returned; recording energy.", flush=True)
        energy = float(np.real(solver.energy))
        row["energy"] = energy
        row["energy_per_site"] = energy / args.L
        finish_record(row, started, "COMPLETED", monitor=monitor)
        print("Limit record finalized.", flush=True)
    except KeyboardInterrupt:
        finish_record(row, started, "INTERRUPTED", "Stopped by keyboard interrupt.", monitor)
        append_csv("dmrg_limit_search.csv", row)
        raise
    except Exception as exc:
        finish_record(row, started, "ERROR", f"{type(exc).__name__}: {exc}", monitor)
    append_csv("dmrg_limit_search.csv", row)
    return row


def run_vqe(args) -> dict:
    import tensorcircuit as tc
    import tensorflow as tf

    tc.set_backend("tensorflow")
    tc.set_dtype("complex64")
    started = time.perf_counter()
    row = base_record("VQE/statevector", args.L, args.g)
    row["tensorcircuit_version"] = getattr(tc, "__version__", installed_version("tensorcircuit"))
    row["tensorflow_version"] = tf.__version__
    row["tensorflow_devices"] = "; ".join(
        f"{device.device_type}:{device.name}"
        for device in tf.config.list_physical_devices()
    )
    statevector_mib = math.inf if args.L > 1039 else math.ldexp(1.0, args.L - 17)
    working_estimate_gib = math.inf if args.L > 1042 else math.ldexp(1.0, args.L - 19)
    hilbert_dimension = 2**args.L if args.L <= 1023 else f"2^{args.L}"
    row.update({
        "hilbert_dimension": hilbert_dimension,
        "layers": args.layers,
        "steps_requested": args.steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "complex64_statevector_lower_bound_mib": statevector_mib,
        "working_set_estimate_gib_heuristic": working_estimate_gib,
        "working_set_estimate_basis": (
            "256x complex64 statevector; calibrated from measured L=20 working set"
        ),
        "time_budget_seconds": args.budget_seconds,
    })

    if (args.L < 2 or args.layers < 1 or args.steps < 1
            or args.budget_seconds <= 0 or args.min_free_gib < 0):
        row["status"] = "SKIPPED_INVALID_CONFIGURATION"
        append_csv("vqe_limit_search.csv", row)
        return row
    reserve_gib = max(args.min_free_gib, ABSOLUTE_MIN_FREE_RAM_RESERVE_GIB)
    row["minimum_free_ram_reserve_gib"] = reserve_gib
    if row["available_gib_before"] - working_estimate_gib < reserve_gib:
        row["status"] = "SKIPPED_MEMORY_RESERVE"
        append_csv("vqe_limit_search.csv", row)
        return row

    zz = np.kron(tc.gates._z_matrix, tc.gates._z_matrix)

    def energy(params):
        circuit = tc.Circuit(args.L)
        params_cast = tc.backend.cast(params, tc.dtypestr)
        for site in range(args.L):
            circuit.H(site)
        for layer in range(args.layers):
            for site in range(args.L - 1):
                circuit.exp1(site, site + 1, unitary=zz,
                             theta=params_cast[2 * layer, site])
            for site in range(args.L):
                circuit.rx(site, theta=params_cast[2 * layer + 1, site])
        # Aligned with Quimb's j=+1 antiferromagnetic spin-1/2 TFIM:
        # H = +(J/4) sum Z_i Z_{i+1} - (h/2) sum X_i, with J=1 and h=g.
        value = 0.0
        for site in range(args.L):
            value += -(args.g / 2.0) * circuit.expectation((tc.gates.x(), [site]))
        for site in range(args.L - 1):
            value += +0.25 * circuit.expectation(
                (tc.gates.z(), [site]), (tc.gates.z(), [site + 1])
            )
        return tc.backend.real(value)

    trace = []
    trace_path = RESULTS / (
        f"limit_trace_vqe_L{args.L}_g{args.g}_d{args.layers}_"
        f"s{args.seed}_{int(time.time())}.csv"
    )

    def save_trace() -> None:
        if not trace:
            return
        with trace_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "energy"])
            writer.writerows(enumerate(trace, start=1))
        row["trace_file"] = trace_path.name

    monitor = ResourceMonitor()
    monitor.peak_rss_mib = row["rss_before_mib"]
    monitor.minimum_available_gib = row["available_gib_before"]
    try:
        monitor.start()
        tf.keras.utils.set_random_seed(args.seed)
        params = tf.Variable(tf.random.normal([2 * args.layers, args.L], stddev=0.1))
        optimizer = tf.keras.optimizers.Adam(args.learning_rate)
        optimization_started = time.perf_counter()
        status = "COMPLETED"
        for step in range(args.steps):
            with tf.GradientTape() as tape:
                value = energy(params)
            gradient = tape.gradient(value, params)
            if gradient is None:
                raise RuntimeError("TensorFlow returned a missing gradient.")
            optimizer.apply_gradients([(gradient, params)])
            trace.append(float(value.numpy()))
            if time.perf_counter() - optimization_started >= args.budget_seconds:
                status = "TIME_BUDGET_STOPPED"
                break

        row["steps_completed"] = len(trace)
        row["energy"] = trace[-1] if trace else ""
        finish_record(row, started, status, monitor=monitor)
        save_trace()
    except KeyboardInterrupt:
        row["steps_completed"] = len(trace)
        row["energy"] = trace[-1] if trace else ""
        finish_record(row, started, "INTERRUPTED", "Stopped by keyboard interrupt.", monitor)
        save_trace()
        append_csv("vqe_limit_search.csv", row)
        raise
    except Exception as exc:
        row["steps_completed"] = len(trace)
        row["energy"] = trace[-1] if trace else ""
        finish_record(row, started, "ERROR", f"{type(exc).__name__}: {exc}", monitor)
        save_trace()
    append_csv("vqe_limit_search.csv", row)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Run one guarded DMRG or VQE stress point and append its record."
    )
    subparsers = parser.add_subparsers(dest="method", required=True)

    d = subparsers.add_parser("dmrg")
    d.add_argument("--L", type=int, default=120)
    d.add_argument("--g", type=float, default=0.5)
    d.add_argument("--chi", type=int, default=16)
    d.add_argument("--max-sweeps", type=int, default=8,
                   help="DMRG sweep budget; set explicitly for reproducibility")
    d.add_argument("--tolerance", type=float, default=1e-8)
    d.add_argument("--min-free-gib", type=float, default=DEFAULT_FREE_RAM_RESERVE_GIB)
    d.set_defaults(run=run_dmrg)

    v = subparsers.add_parser("vqe")
    v.add_argument("--L", type=int, default=14)
    v.add_argument("--g", type=float, default=0.5)
    v.add_argument("--layers", type=int, default=2)
    v.add_argument("--steps", type=int, default=100)
    v.add_argument("--learning-rate", type=float, default=0.01)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--budget-seconds", type=float, default=120.0,
                   help="Per-case wall-clock budget checked after each full optimizer step")
    v.add_argument("--min-free-gib", type=float, default=DEFAULT_FREE_RAM_RESERVE_GIB)
    v.set_defaults(run=run_vqe)

    args = parser.parse_args()
    result = args.run(args)
    print(result)


if __name__ == "__main__":
    main()