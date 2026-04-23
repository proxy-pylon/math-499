"""
Run the N=4 VQLS scaling experiment (Experiment 2 ODE) and save scaling_N4.png.
Uses the entangled ansatz from vqls.py (n_layers=3).
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from collocation import (
    build_collocation_matrix,
    minimize_pauli_l1,
    minimize_pauli_l1_monic,
    count_pauli_strings,
    equidistant_interior_points,
)
from vqls import run_vqls

# ── ODE parameters (Experiment 2) ────────────────────────────────────────────
ALPHA, BETA, GAMMA = 1, 1, 1
T_L, T_R = -1.0, 1.0
U_L, U_R = 0.0, 2.0
f_rhs = lambda t: t**2 + 3*t + 3

N          = 4
N_INTERIOR = N - 2
N_QUBITS   = int(round(np.log2(N)))
STEPS      = 50
TRIALS     = 3
SEEDS      = [0, 1, 2]

_COLORS     = {"Monomial (Case A)": "#e74c3c", "Monic LP": "#27ae60", "Full LP": "#2980b9"}
_LINESTYLES = {"Monomial (Case A)": "-",       "Monic LP": "--",       "Full LP": ":"}

# ── Build b vector ────────────────────────────────────────────────────────────
col_pts = equidistant_interior_points(N_INTERIOR, T_L, T_R)
b_raw   = np.array([U_L, U_R] + [f_rhs(t) for t in col_pts], dtype=float)
b_norm  = b_raw / np.linalg.norm(b_raw)

# ── Collect matrices ──────────────────────────────────────────────────────────
M_A     = build_collocation_matrix(N_INTERIOR, ALPHA, BETA, GAMMA, T_L, T_R)
M_monic, _, _ = minimize_pauli_l1_monic(N_INTERIOR, ALPHA, BETA, GAMMA, T_L, T_R)
M_lp,   _, _ = minimize_pauli_l1(N_INTERIOR, ALPHA, BETA, GAMMA, T_L, T_R)

cases = [
    ("Monomial (Case A)", M_A),
    ("Monic LP",          M_monic),
    ("Full LP",           M_lp),
]

# ── Run VQLS ──────────────────────────────────────────────────────────────────
results = {}
for name, M_raw in cases:
    n_paulis, _ = count_pauli_strings(M_raw)
    M_norm = M_raw / np.linalg.norm(M_raw, "fro")
    print(f"\n{'='*55}")
    print(f"  {name}  (L={n_paulis}/{4**N_QUBITS})")
    print(f"{'='*55}")
    hists = []
    for seed in SEEDS:
        print(f"  seed={seed} ...", flush=True)
        hist, elapsed, _ = run_vqls(M_norm, b_norm, steps=STEPS,
                                     rng_seed=seed, verbose=True)
        hists.append(hist)
        print(f"  done in {elapsed:.1f}s, final cost={hist[-1]:.5f}")
    results[name] = {"n_paulis": n_paulis, "histories": hists}

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  SUMMARY — N={N}, {STEPS} steps, {TRIALS} seeds")
print(f"{'='*55}")
for name, r in results.items():
    finals = [h[-1] for h in r["histories"]]
    print(f"  {name:<22}  L={r['n_paulis']}  "
          f"cost@{STEPS}: {np.mean(finals):.4f} ± {np.std(finals):.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, r in results.items():
    color = _COLORS[name]
    ls    = _LINESTYLES[name]
    hists = np.array(r["histories"])
    mean_h = hists.mean(axis=0)
    steps_ax = np.arange(1, STEPS + 1)
    for h in hists:
        ax.plot(steps_ax, h, color=color, alpha=0.2, linewidth=0.7, linestyle=ls)
    ax.plot(steps_ax, mean_h, color=color, linewidth=2.0, linestyle=ls,
            label=f"{name}  (L={r['n_paulis']})")

ax.set_xlabel("VQLS step", fontsize=12)
ax.set_ylabel("Local cost", fontsize=12)
ax.set_title(f"VQLS convergence — N={N} ({N_QUBITS} qubits, Exp 2 ODE)", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(1, STEPS)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out = os.path.join(ROOT_DIR, "plots", "scaling_N4.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nSaved → {out}")
