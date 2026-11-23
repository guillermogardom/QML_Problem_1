# Scope and Evidence Map

This study directly fulfills the 2026–2027 session instructions: reproduce the Quimb DMRG/MPS/TFIM and VQE/TFIM tutorial plots locally, vary physical and algorithmic parameters, scale the spin/qubit count until practical limits are exposed, document accuracy and failure modes, and deliver a clean English lab report in a shareable repository.

## Evidence in This Repository

* **Reference baselines:** Unmodified instructor notebooks are preserved under `notebooks/tutorial_source/`, alongside the course theory notes in `references/notes_ising.pdf`.
* **DMRG/MPS stress tests (`notebooks/01_DMRG_MPS_Stress_Tests.ipynb`):** Reproduces the tutorial's 51-point $L=40$ field scan, $\chi$ convergence (energy and mid-chain entropy $S(L/2)$), chain-length scaling ($L=20\dots 200$), bipartite entanglement entropy, spatial correlations, critical-exponent diagnostic fits ($\beta, c, \eta$), and dense exact-diagonalization checks ($L=4\dots 10$).
* **VQE stress tests (`notebooks/02_VQE_TFIM_Stress_Tests.ipynb`):** Retains the tutorial circuit ansatz and Adam workflow while scaling $L=4\dots 12$ across multiple random seeds, varying transverse field $g$, circuit depth ($1..4$ layers), learning rate, and step budget, alongside Hilbert-space memory lower bounds.
* **Frontier stress runner (`limit_search.py`):** Tests larger $L$ and $\chi$ values one case at a time with $0.2\text{ s}$ RSS/RAM sampling and a protective memory preflight.
* **Synthesis & report (`notebooks/03_Lab_Report_and_Comparison.ipynb`, `LAB_REPORT.md`, `MODEL_EXPLANATION.tex`):** Integrates all aligned results from `results/tutorial_faithful/` and distinguishes tested frontiers from hardware failures.

## Tutorial Discrepancies Resolved

1. **DMRG analytic order-parameter curve:** The source notebook fixed `j=1` in Quimb but evaluated `lambda = b_x / (1 - b_x)`. Our extension plots both the literal tutorial curve and the exact Pfeuty formula $[1 - (2g)^2]^{1/8}$ for fixed $j=1$.
2. **VQE Hamiltonian alignment:** The source notebook used `h=1` in `exact_energy_tf`, `bx=0.5` in `dmrg_energy`, and negative $ZZ$ in `vqe_energy`. Our aligned code standardizes all three methods to the same field ratio $g = h/J$ and positive $+J/4$ $ZZ$ coupling.
3. **Synthetic GPU block:** The tutorial's JAX GPU section measured synthetic random-matrix SVDs rather than Quimb DMRG and is omitted on our CPU-only test machine.