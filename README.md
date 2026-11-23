# QML Problem 1 — Tutorial-Aligned TFIM Study

This repository extends the 2026–2027 Quimb DMRG/MPS/TFIM and TensorCircuit VQE/TFIM tutorials with systematic parameter sweeps, chain-length and qubit scaling, stability/accuracy checks, controlled hardware-frontier measurements, and a complete lab report.


## Tutorial Fixes & Methodological Alignment

1. **VQE Hamiltonian alignment (`02_VQE_TFIM_Stress_Tests.ipynb` & `limit_search.py`):**
   * In the original `VQE_TFIM.ipynb` tutorial, `exact_energy_tf` passed `h=1.0` (`bx=1.0`, paramagnetic), `dmrg_energy` passed `bx=0.5` (critical), and `vqe_energy` used a negative $ZZ$ coefficient (`-J/4`).
   * The aligned repository standardizes all three methods to the same field ratio $g = h/J$ and positive antiferromagnetic coupling (`+J/4` for $ZZ$, matching Quimb's `j=+1` spin-$1/2$ convention), keeping the tutorial's Hadamard/$ZZ$/$R_x$ circuit ansatz and Adam optimizer intact.
2. **Order-parameter analytic reference (`01_DMRG_MPS_Stress_Tests.ipynb`):**
   * For fixed `j=1.0` in Quimb ($H = +\sum S_i^z S_{i+1}^z - b_x \sum S_i^x$), the exact thermodynamic longitudinal order parameter is $[1 - (2 b_x / J)^2]^{1/8}$ for $b_x/J < 0.5$ and $0$ above (Pfeuty, 1970).
   * The original tutorial coded `lambda = b_x / (1 - b_x)`. Our notebook plots **both** the fixed-$H$ Pfeuty reference and the literal course formula so the parameterization discrepancy is directly inspectable.
3. **Observable evaluation on Windows:**
   * Full tensor-network contraction (`psi.H @ psi_z`) triggered a `UnicodeEncodeError` in some local Windows environments. The aligned DMRG notebook uses Quimb's MPS-native `local_expectation` method (with a safe fallback) to evaluate $\langle Z_{L/2} \rangle$ reliably.
4. **CPU-only scope (GPU section omitted):**
   * The local machine exposes CPU only (`CPU:0`), and the GPU section in the original DMRG tutorial benchmarked synthetic random-matrix SVDs in JAX rather than actual Quimb DMRG sweeps. All benchmarks in this repository measure real CPU execution, and no GPU speedup claims are made.

## Repository Contents

* `notebooks/tutorial_source/`: Verbatim instructor notebooks preserved with original outputs.
* `notebooks/01_DMRG_MPS_Stress_Tests.ipynb`: 51-point field scan ($L=40$, $\chi_{\max}=64$), $\chi$ convergence (energy, wall time, and central entanglement entropy), chain-length scaling up to $L=200$, bipartite entanglement entropy, correlations, critical-exponent fits ($\beta, c, \eta$), and exact-diagonalization cross-checks ($L \le 10$).
* `notebooks/02_VQE_TFIM_Stress_Tests.ipynb`: Aligned Hadamard/$ZZ$/$R_x$ VQE with repeated-seed size scaling ($L=4..12$), field sweeps ($g \in [0.1, 1.0]$), circuit-depth sweeps ($1..4$ layers), optimizer grid (steps $\times$ learning rate), and statevector memory scaling bounds.
* `notebooks/03_Lab_Report_and_Comparison.ipynb`: Post-processing and comparative figures generated strictly from `results/tutorial_faithful/`.
* `limit_search.py`: Guarded single-case stress runner for pushing $L$ and $\chi$ toward practical memory/time frontiers while sampling RSS and available RAM every $0.2\text{ s}$.
* `LAB_REPORT.md` & `MODEL_EXPLANATION.tex`: Integrated quantitative lab report and theoretical background.
* `docs/`: Crosswalk, machine/environment record, and reproducibility protocol.

## Reproduction

Use the project `.venv` as the Python/Jupyter kernel. From a fresh checkout:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt