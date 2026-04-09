# Tiny Language Model Based Context-Aware Resource Allocation for 6G Semantic Communication Systems

## Overview

This project implements a semantic-aware resource allocation framework for 6G wireless systems, where each user message is assigned an importance score using a lightweight TinyML model. The system allocates power based on both channel conditions and semantic importance, while ensuring queue stability under stochastic arrivals.

The core objective is not to maximize raw throughput, but to improve **delay, stability, and fairness under heterogeneous load conditions**.

---

## Key Idea

Traditional schedulers allocate resources based only on channel quality (e.g., MAX-CSI) or fairness (e.g., Round-Robin).

This work introduces:

* **Semantic weighting** → prioritize important messages
* **Lyapunov Drift-Plus-Penalty (DPP)** → control queue stability
* **Semantic Water-Filling** → closed-form optimal power allocation

---

## Method Summary

1. **Semantic Scoring (TinyML)**

   * Each message is assigned a score ( S_i \in [0,1] )
   * Treated as fixed input per time slot

2. **Lyapunov Control**

   * Objective:
     [
     \min -V \sum_i S_i R_i + \sum_i Q_i(\lambda_i - \mu_i)
     ]

3. **Semantic Water-Filling**

   * Closed-form solution:
     [
     P_i^* = \left[\frac{S_i}{\mu} - \frac{N_0 B}{|h_i|^2}\right]^+
     ]

---

## Results (Key Takeaways)

* ~1.8× delay reduction under heterogeneous load
* Strong reduction in queue variance
* No artificial gains in normal regime
* 19× compute energy reduction via QAT

---

## Repository Structure

```bash
semantic-resource-allocation-6g/
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── pdf/
│       └── final_paper.pdf
│
├── code/
│   └── Simulation.py
│
├── results/
│   ├── stress_scenario.png
│   ├── queue_evolution.png
│   ├── tail_delay_distribution.png
│   ├── utility_comparison.png
│   └── alpha_sensitivity.png
│
├── README.md
└── requirements.txt

```

---

## How to Run (Mac/Linux)

### 1. Navigate to project

```bash
cd semantic-resource-allocation-6g
```

---

### 2. Create virtual environment

```bash
python3 -m venv .venv
```

---

### 3. Activate environment

```bash
source .venv/bin/activate
```

---

### 4. Ensure pip is available

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

---

### 5. Install dependencies

```bash
pip install numpy matplotlib scipy
```

(**Note:** Using `requirements.txt` may fail on newer Python versions due to build issues. Direct install ensures compatibility.)

---

### 6. Run simulation

```bash
python code/Simulation.py
```

---

## Output

After successful execution:

* All experiment results printed in terminal
* Final figure saved at:

```bash
results/6G_TinyML_final.png
```

---

## What This Work Proves

* Semantic awareness alone is insufficient
* Queue-awareness (Lyapunov) is dominant under load
* Gains appear under stress, not ideal conditions
* This is a robustness contribution

---

## Limitations

* Semantic scores assumed fixed per slot
* Single base station setup
* Simulation only (no hardware validation)

---

## References

Core theoretical foundations:

* Neely (2010) — Lyapunov optimization
* Boyd (2011) — ADMM
* Razaviyayn (2013) — SCA
* Rosen (1965) — Nash equilibrium
* Ma et al. (2024) — BitNet

---

## License

For academic and research use only.
