# Transverse-Field Ising Model: DMRG/MPS and VQE Stress-Test Report

## 1. Executive Summary

This report documents a controlled computational and physical comparison of two ground-state solvers for the open-boundary 1D Transverse-Field Ising Model (TFIM), extending the 2026–2027 Quimb DMRG/MPS and TensorCircuit VQE tutorials:

* **DMRG/MPS (Quimb CPU):** All **58/58** structured sweep runs and **51/51** field-scan points completed cleanly. Against dense exact diagonalization ($L \le 10$, $\chi=32$), DMRG reproduces ground-state energies to within **`3.578e-08`** absolute error. Chain length scales smoothly to **$L=200$** ($\chi=32$ in **`8.51 s`**, **`209.0 MiB`** peak RSS); the practical bottleneck is $O(\chi^3)$ SVD runtime at large bond dimension (**`133.22 s`** at $L=40, \chi=256$ with no further energy gain over $\chi=128$).
* **VQE Statevector (TensorCircuit + TensorFlow CPU):** All **81/81** structured runs completed cleanly across size, seed, field, depth, and optimizer sweeps. At fixed depth ($2$ layers) and $300$ Adam steps ($\mathrm{lr}=0.01$), mean absolute energy error grows from **`0.00192`** at $L=4$ to **`0.09788`** at $L=12$. In guarded frontier tests (`limit_search.py`), statevector VQE scales up to **$L=20$ qubits** (**`44.11 s`** for 15 steps, **`1868.4 MiB`** peak RSS), while **$L=22$** is blocked by the **`1.75 GiB`** RAM safety reserve (`SKIPPED_MEMORY_RESERVE` against `4.04 GiB` available RAM), demonstrating the exponential $O(2^L)$ memory wall.

---

## 2. Hamiltonian Convention & Tutorial Alignment Fixes

For spin-$1/2$ operators ($S_i^\alpha = \sigma_i^\alpha / 2$), all solvers in this repository evaluate the exact same antiferromagnetic open-chain Hamiltonian:

$$H = +J\sum_{i=1}^{L-1} S_i^z S_{i+1}^z - h\sum_{i=1}^{L} S_i^x
= +\frac{J}{4}\sum_{i=1}^{L-1} Z_i Z_{i+1} - \frac{h}{2}\sum_{i=1}^{L} X_i, \quad g = \frac{h}{J}, \quad g_c = 0.5.$$

Three discrepancies in the source tutorials were explicitly resolved and documented:
1. **VQE Hamiltonian alignment:** The original `VQE_TFIM.ipynb` mixed `h=1.0` in exact diagonalization, `bx=0.5` in DMRG, and negative $ZZ$ coupling (`-J/4`) in the VQE cost function. Our code aligns Exact, DMRG, and VQE to the same field ratio $g = h/J$ and positive `+J/4` $ZZ$ coupling while keeping the tutorial's Hadamard/$ZZ$/$R_x$ circuit and Adam optimizer intact.
2. **Order-parameter analytic overlay:** The DMRG tutorial fixed `j=1.0` in Quimb while plotting `lambda = b_x / (1 - b_x)`. Our 51-point field plot retains both the literal course formula and the exact Pfeuty thermodynamic order parameter $[1 - (2g)^2]^{1/8}$ ($g < 0.5$, zero for $g \ge 0.5$) for transparent comparison.
3. **Windows observable contraction & CPU-only scope:** Replaced `psi.H @ psi_z` with Quimb's MPS-native `local_expectation` (preventing Windows `UnicodeEncodeError`), and omitted the tutorial's synthetic JAX random-SVD GPU block because the test machine exposes `CPU:0` only.

---

## 3. Experimental Environment

* **OS & Hardware:** Windows 10 (build 19045), Intel64 Family 6 Model 186 Stepping 2 (12 physical / 16 logical CPUs), 15.72 GiB installed RAM, CPU-only TensorFlow (`CPU:0`).
* **Software Stack:** Python 3.12.2, `quimb==1.16.0`, `tensorcircuit==1.10.0`, `tensorflow==2.21.0`, `numpy==2.5.5`, `scipy==1.18.1`, `pandas==3.1.1`, `matplotlib==3.11.2`, `psutil==7.2.2`.

---

## 4. DMRG/MPS Results (`01_DMRG_MPS_Stress_Tests.ipynb`)

### 4.1 Exact-Diagonalization Validation ($L \le 10, \chi=32$)
Across all 12 combinations of $L \in \{4, 6, 8, 10\}$ and $g \in \{0.20, 0.50, 0.80\}$, DMRG reproduces dense exact-diagonalization ground-state energies with a maximum absolute error of **`3.578e-08`** (consistent with `tol=1e-8`).

### 4.2 Bond-Dimension Convergence ($L=40$)
| Regime | $g$ | $E/L$ ($\chi=8$) | $E/L$ ($\chi=128$) | $S(L/2)$ ($\chi=128$) | Runtime $\chi=8 \to 128$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ordered low field | 0.20 | `-0.25436493` | `-0.25436493` | `0.001719` | `0.22 s` $\to$ `10.60 s` |
| Near critical | 0.50 | `-0.31605888` | `-0.31605896` | `0.728903` | `0.40 s` $\to$ `11.84 s` |
| High field | 0.80 | `-0.43906397` | `-0.43906397` | `0.194833` | `0.17 s` $\to$ `10.77 s` |

Energy per site saturates by $\chi=16$ at $L=40$ ($|\Delta E/L| \approx 7.52 \times 10^{-8}$ from $\chi=8 \to 16$ at criticality, and $\sim 10^{-13}$ thereafter), while runtime grows sharply due to $O(\chi^3)$ SVD cost.

### 4.3 Chain-Length Scaling ($L=20\dots 200, \chi=16$)
All 24 scaling cases completed cleanly (`dmrg_size_scaling.png`). At $L=200$, runtimes remain low: **`2.85 s`** ($g=0.2$), **`6.98 s`** ($g=0.5$), and **`2.80 s`** ($g=0.8$), with post-run RSS snapshots stable around **`319–331 MiB`**. At $L=200, g=0.5$, $E/L = -0.317856$, approaching the exact thermodynamic critical density $-1/\pi \approx -0.318310$.

### 4.4 Entanglement, Correlations, and Critical-Exponent Diagnostics
* **Central charge ($L=60, \chi=64$, margin $=6$):** Fitting the Calabrese-Cardy CFT formula in base-2 bits yields **$c = 0.51834 \pm 0.00043$**, within $3.7\%$ of the exact Ising CFT value $c = 0.50$.
* **Correlation exponent ($L=80, \chi=64$):** Power-law fits to $|\langle Z_i Z_{i+r}\rangle|$ at $g=0.5$ yield $\eta = 0.2920$ ($r \in [2,8]$), $\eta = 0.3193$ ($r \in [3,12]$), and $\eta = 0.3596$ ($r \in [5,16]$) versus exact $\eta = 0.25$. Drift with $r_{\max}$ reflects open-boundary finite-size effects.
* **Order-parameter exponent ($L=40$, 51-point scan):** Fitting $\log|m_z|$ on $g \in (0.40, 0.47)$ ($n=3$ points) produces $\beta = 6.628 \pm 1.743$ versus exact $\beta = 0.125$. Without an explicit symmetry-breaking pinning field, finite-size $Z_2$ symmetry restoration suppresses $|m_z|$ well before $g_c=0.5$, rendering a naive finite-chain $\beta$ fit unreliable.

---

## 5. VQE Results (`02_VQE_TFIM_Stress_Tests.ipynb`)

### 5.1 Critical-Field Size & Seed Scaling ($g=0.5$, 2 layers, 300 Adam steps, $\mathrm{lr}=0.01$)
| $L$ | $\dim(\mathcal{H})$ | Reference $E_0$ (Kind) | Mean $E_{\mathrm{VQE}}$ | Mean $|\Delta E|$ (Range) | Mean Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | 16 | `-1.189693` (Exact) | `-1.187774` | `0.001918` (`0.00079–0.00353`) | `16.11` |
| 6 | 64 | `-1.824057` (Exact) | `-1.799462` | `0.024595` (`0.02085–0.02945`) | `25.86` |
| 8 | 256 | `-2.459488` (Exact) | `-2.403073` | `0.056415` (`0.04383–0.06720`) | `34.78` |
| 10 | 1,024 | `-3.095372` (Exact) | `-3.024753` | `0.070619` (`0.06097–0.07931`) | `44.70` |
| 12 | 4,096 | `-3.731493` (DMRG $\chi=64$) | `-3.633617` | `0.097876` (`0.08701–0.11872`) | `61.12` |

At fixed depth (2 layers), error rises monotonically with $L$ because shallow nearest-neighbour layers have a restricted correlation light-cone across critical chains, and optimization difficulty increases with parameter count ($2 \times \text{layers} \times L$).

### 5.2 Field, Depth, and Optimizer Sensitivity ($L=8$, `vqe_parameter_sweeps.png`)
* **Field dependence ($g \in [0.10, 1.00]$):** Error is largest in the ordered antiferromagnetic regime ($g=0.10 \to |\Delta E| \approx 0.24$) and drops monotonically toward the paramagnetic limit ($g=1.00 \to |\Delta E| \approx 0.01$). This reflects the circuit initialization: Hadamards prepare $|+\rangle^{\otimes L}$, which is already the exact $g \to \infty$ ground state, whereas building long-range antiferromagnetic $Z$-order requires greater entanglement depth.
* **Depth dependence ($g=0.5$, 300 steps):** Increasing circuit depth from 1 to 4 layers reduces mean absolute error from **`~0.092`** (1 layer) $\to$ **`~0.056`** (2 layers) $\to$ **`~0.025`** (3 layers) $\to$ **`~0.022`** (4 layers), where accuracy begins to plateau for this optimizer budget.
* **Optimizer grid ($g=0.5$, 2 layers):** `lr=0.001` severely under-converges (`0.391` at 100 steps, `0.159` at 300, `0.086` at 600). Raising the learning rate to `0.05` improves convergence across all budgets (`0.055` at 100 steps, `0.031` at 300, and **`0.025`** at 600 steps, the best sampled setting).

---

## 6. Resource Frontiers & Breaking Points (`limit_search.py`)

Single-case tests with $0.2\text{ s}$ RSS/RAM sampling (`limit_frontier.png`) expose fundamentally different scaling walls:

| Method | Configuration | Status | Wall Time (s) | Sampled Peak RSS (MiB) | Energy $E$ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| DMRG/MPS | $L=120, \chi=16$ | `COMPLETED` | `3.49` | `179.4` | `-38.106612` |
| DMRG/MPS | $L=200, \chi=32$ | `COMPLETED` | `8.51` | `209.0` | `-63.571293` |
| DMRG/MPS | $L=40, \chi=128$ | `COMPLETED` | `18.21` | `211.1` | `-12.642358` |
| DMRG/MPS | $L=40, \chi=256$ | `COMPLETED` | `133.22` | `332.9` | `-12.642358` |
| VQE | $L=14$, 100 steps | `COMPLETED` | `40.31` | `510.7` | `-4.181875` |
| VQE | $L=16$, 50 steps | `COMPLETED` | `26.38` | `587.5` | `-4.730271` |
| VQE | $L=18$, 30 steps | `COMPLETED` | `28.81` | `853.0` | `-5.072601` |
| VQE | $L=20$, 15 steps | `COMPLETED` | `44.11` | `1868.4` | `-5.262589` |
| VQE | $L=22$, 10 steps | `SKIPPED_MEMORY_RESERVE` | `0.00` | — | — |

* **Where DMRG breaks down:** Not in RAM (`333 MiB` even at $\chi=256$), but in **CPU wall time**, which scales as $O(\chi^3)$ ($18.2\text{ s} \to 133.2\text{ s}$ from $\chi=128 \to 256$). For 1D gapped/critical chains, $L$ scales only linearly.
* **Where Statevector VQE breaks down:** In **RAM and per-step differentiation time**, both scaling exponentially as $O(2^L)$. Peak RSS more than doubles from $L=18$ (`853 MiB`) to $L=20$ (`1868 MiB`, ~`2.94 s/step`). At $L=22$ ($2^{22} \approx 4.19 \times 10^6$ amplitudes), the working set exceeded the `1.75 GiB` safety reserve against `4.04 GiB` available RAM and was cleanly halted by preflight protection.

---

## 7. Limitations & Conclusion

* **Limitations:** Ideal classical statevector VQE omits hardware noise, finite-shot statistical variance, and device connectivity; `0.2 s` RSS sampling can miss brief intermediate tensor allocation spikes; and finite open-chain DMRG without symmetry-breaking fields cannot directly resolve the thermodynamic $\beta$ exponent.
* **Conclusion:** For 1D TFIM ground states, DMRG/MPS achieves machine-precision energies (`~3.6e-8` vs exact) and scales effortlessly to hundreds of spins in seconds, whereas classical statevector VQE is variationally constrained at shallow depth (especially in the antiferromagnetic phase) and hits a hard exponential memory wall around $20\text{–}22$ qubits on a $16\text{ GiB}$ CPU workstation.