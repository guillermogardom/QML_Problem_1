# Environment and Machine Record

The archived negative-$j$ experiments in the top-level `results/` directory used the project-local `.venv` (64-bit Python 3.12.2). The tutorial-aligned campaign writes fresh environment snapshots directly into:

* `results/tutorial_faithful/dmrg_environment_record.json`
* `results/tutorial_faithful/vqe_environment_record.json`

## Recorded Local Machine

* **Operating system:** Windows 10 (build 19045).
* **CPU reported by Python:** Intel64 Family 6, Model 186, Stepping 2; 16 logical CPUs.
* **Installed RAM:** 15.72 GiB.
* **TensorFlow compute devices:** `CPU:0` only (no GPU device exposed).

## Software Stack

Exact pinned package versions are specified in `requirements.txt` (including `numpy`, `scipy`, `matplotlib`, `pandas`, `psutil`, `quimb`, `tensorcircuit`, and `tensorflow`). Exact diagonalization in `01_DMRG_MPS_Stress_Tests.ipynb` and `02_VQE_TFIM_Stress_Tests.ipynb` uses Quimb's `.to_dense()` operator representation so `qibo` is not required as an extra dependency.

## Environment Handling

The `.venv` directory is excluded via `.gitignore` because it contains machine-specific paths. Recreate the environment with `python -m pip install -r requirements.txt`. All timings and memory traces correspond strictly to CPU execution.