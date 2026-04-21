"""
experiment_scaling.py

Scaling experiment for the ODE from Experiment 2:
    u'' + u' + u = t^2 + 3t + 3  on [-1, 1],  u(-1)=0, u(1)=2
    Exact solution: u(t) = t^2 + t

For N = 4, 8, 16 and three basis choices:
  1. Monomial (monic) basis  — Case A, C = I (standard t^k basis)
  2. Monic LP               — in-between, lower-triangular C with unit diagonal
  3. Full LP                — global optimum (always M* = I_N, 1 Pauli string)

Measurements per (N, basis):
  - Number of Pauli terms in the decomposition
  - Condition number of the collocation matrix
  - LP preprocessing time: mean ± std of 3 trials (N/A for monomial)
  - VQLS time for 25 steps: mean ± std of 3 trials (seeds 0, 1, 2)
  - VQLS cost after 25 steps: mean ± std of 3 trials

Outputs (created relative to this file's parent directory):
  logs/scaling_N{N}.log        — per-N detailed log
  logs/scaling_summary.txt     — combined summary table
  plots/scaling_N{N}.png       — error vs VQLS step (all 3 bases on one plot)
"""

import sys
import os
import time
import logging
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

LOGS_DIR  = os.path.join(ROOT_DIR, "logs")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")

from collocation import (
    build_collocation_matrix,
    minimize_pauli_l1,
    minimize_pauli_l1_monic,
    count_pauli_strings,
    equidistant_interior_points,
)

# vqls.py has a hardcoded sys.path.insert for CODE_DIR, which is the same directory.
from vqls import run_vqls

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALPHA, BETA, GAMMA = 1, 1, 1
T_L, T_R           = -1.0, 1.0
U_L, U_R           = 0.0, 2.0

def f_rhs(t):
    return t**2 + 3*t + 3

N_VALUES       = [4, 8, 16]
VQLS_STEPS     = 25
N_TRIALS       = 3
VQLS_SEEDS     = [0, 1, 2]

# Skip VQLS when forward-pass Hadamard-test estimate exceeds this.
# Forward pass per step  ≈  L^2 * (n_qubits + 1).
# (Does not include gradient overhead, so actual cost is ~(1 + 2n) × this.)
VQLS_SKIP_THRESHOLD = 30_000

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_dirs():
    os.makedirs(LOGS_DIR,  exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def make_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_b_raw(n_interior: int) -> np.ndarray:
    """RHS vector [u_L, u_R, f(t_1), ..., f(t_k)] (not yet normalised)."""
    col_pts = equidistant_interior_points(n_interior, T_L, T_R)
    return np.array([U_L, U_R] + [f_rhs(t) for t in col_pts], dtype=float)


def time_lp(func, *args, n_trials: int = 3):
    """Run func(*args) n_trials times.  Returns (mean_s, std_s, last_result)."""
    times = []
    last  = None
    for _ in range(n_trials):
        t0   = time.perf_counter()
        last = func(*args)
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std(ddof=0)), last


def ht_per_step(n_paulis: int, n_qubits: int) -> int:
    """Forward-pass Hadamard-test estimate per VQLS step: L^2 * (n+1)."""
    return n_paulis**2 * (n_qubits + 1)


# ---------------------------------------------------------------------------
# VQLS runner (multiple trials)
# ---------------------------------------------------------------------------

def run_vqls_trials(M_raw: np.ndarray, b_raw: np.ndarray,
                    n_trials: int = 3, seeds=None, steps: int = 25,
                    verbose: bool = False):
    """Run VQLS `n_trials` times with different RNG seeds.

    Returns
    -------
    mean_time, std_time, mean_cost, std_cost, cost_histories
    """
    if seeds is None:
        seeds = list(range(n_trials))

    M_norm = M_raw / np.linalg.norm(M_raw, "fro")

    times, final_costs, histories = [], [], []
    for seed in seeds:
        hist, elapsed, _ = run_vqls(M_norm, b_raw, steps=steps,
                                     rng_seed=seed, verbose=verbose)
        times.append(elapsed)
        final_costs.append(hist[-1] if hist else float("nan"))
        histories.append(hist)

    t_arr = np.array(times)
    c_arr = np.array(final_costs)
    return (
        float(t_arr.mean()), float(t_arr.std(ddof=0)),
        float(np.nanmean(c_arr)), float(np.nanstd(c_arr, ddof=0)),
        histories,
    )


# ---------------------------------------------------------------------------
# Per-(N, basis) runner
# ---------------------------------------------------------------------------

def run_one(N: int, basis_name: str, M_raw: np.ndarray, b_raw: np.ndarray,
            logger: logging.Logger, lp_time_mean=None, lp_time_std=None):
    """Collect all metrics for one (N, basis) pair.  Returns a result dict."""
    n_qubits = int(round(np.log2(N)))
    n_paulis, _ = count_pauli_strings(M_raw)
    cond         = np.linalg.cond(M_raw)
    ht           = ht_per_step(n_paulis, n_qubits)
    total_paulis = 4**n_qubits

    logger.info(
        f"  [{basis_name}] Paulis={n_paulis}/{total_paulis}, "
        f"cond={cond:.3e}, HT/step≈{ht:,}"
    )

    result = dict(
        N=N, basis=basis_name,
        n_paulis=n_paulis, total_paulis=total_paulis,
        cond=cond,
        lp_time_mean=lp_time_mean, lp_time_std=lp_time_std,
        vqls_time_mean=None, vqls_time_std=None,
        vqls_cost_mean=None, vqls_cost_std=None,
        cost_histories=[],
        skipped_vqls=False, skip_reason="",
    )

    if ht > VQLS_SKIP_THRESHOLD:
        result["skipped_vqls"] = True
        result["skip_reason"]  = f"HT/step≈{ht:,} > {VQLS_SKIP_THRESHOLD:,}"
        logger.info(f"    VQLS skipped: {result['skip_reason']}")
        return result

    logger.info(f"    Running VQLS — {N_TRIALS} trials × {VQLS_STEPS} steps …")
    mt, st, mc, sc, hists = run_vqls_trials(
        M_raw, b_raw,
        n_trials=N_TRIALS, seeds=VQLS_SEEDS, steps=VQLS_STEPS,
        verbose=True,
    )
    result.update(
        vqls_time_mean=mt, vqls_time_std=st,
        vqls_cost_mean=mc, vqls_cost_std=sc,
        cost_histories=hists,
    )
    logger.info(
        f"    VQLS done: time={mt:.2f}±{st:.2f}s, "
        f"cost@{VQLS_STEPS}={mc:.5f}±{sc:.5f}"
    )
    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_COLORS     = {"Monomial (Case A)": "#e74c3c", "Monic LP": "#27ae60", "Full LP": "#2980b9"}
_LINESTYLES = {"Monomial (Case A)": "-",       "Monic LP": "--",       "Full LP": ":"}


def make_plot(N: int, results: list):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False

    for r in results:
        if r["skipped_vqls"] or not r["cost_histories"]:
            continue

        name     = r["basis"]
        color    = _COLORS.get(name, "gray")
        ls       = _LINESTYLES.get(name, "-")
        hists    = np.array(r["cost_histories"])   # (n_trials, steps)
        mean_h   = hists.mean(axis=0)
        steps_ax = np.arange(1, len(mean_h) + 1)

        # Individual trial curves (faint)
        for h in hists:
            ax.plot(np.arange(1, len(h) + 1), h,
                    color=color, alpha=0.2, linewidth=0.7, linestyle=ls)
        # Mean curve
        ax.plot(steps_ax, mean_h,
                color=color, linewidth=2.0, linestyle=ls,
                label=f"{name}  (L={r['n_paulis']})")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    n_qubits = int(round(np.log2(N)))
    ax.set_xlabel("VQLS step", fontsize=12)
    ax.set_ylabel("Local cost", fontsize=12)
    ax.set_title(f"VQLS convergence — N={N} ({n_qubits} qubits, Exp 2 ODE)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(1, VQLS_STEPS)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, f"scaling_N{N}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def fmt_val(mean, std, unit=""):
    if mean is None:
        return "—"
    return f"{mean:.3f}±{std:.3f}{unit}"


def print_summary(all_results: list, stream=None):
    hdr = (
        f"{'N':>4}  {'Basis':<22}  {'Paulis':>8}  {'Cond':>9}  "
        f"{'LP time (s)':>12}  {'VQLS time (s)':>14}  {'Cost@25':>13}"
    )
    sep = "─" * len(hdr)
    lines = ["", sep, hdr, sep]

    for r in all_results:
        lp_s = fmt_val(r["lp_time_mean"], r["lp_time_std"] or 0.0)

        if r["skipped_vqls"]:
            vqls_s = "skipped"
            cost_s = r["skip_reason"]
        else:
            vqls_s = fmt_val(r["vqls_time_mean"], r["vqls_time_std"] or 0.0)
            cost_s = fmt_val(r["vqls_cost_mean"], r["vqls_cost_std"] or 0.0)

        lines.append(
            f"{r['N']:>4}  {r['basis']:<22}  "
            f"{r['n_paulis']:>4}/{r['total_paulis']:<4}  "
            f"{r['cond']:>9.2e}  "
            f"{lp_s:>12}  {vqls_s:>14}  {cost_s:>13}"
        )

    lines.append(sep)
    text = "\n".join(lines) + "\n"
    print(text)
    if stream is not None:
        stream.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_dirs()
    all_results = []

    for N in N_VALUES:
        n_interior = N - 2
        n_qubits   = int(round(np.log2(N)))

        log_path = os.path.join(LOGS_DIR, f"scaling_N{N}.log")
        log      = make_logger(f"exp_N{N}", log_path)
        log.info("=" * 60)
        log.info(f"Experiment 2 ODE — N={N}  ({n_qubits} qubits, {n_interior} interior pts)")
        log.info(f"u'' + u' + u = t^2+3t+3 on [-1,1], u(-1)=0, u(1)=2")
        log.info("=" * 60)

        b_raw = build_b_raw(n_interior)

        # ── 1. Monomial (Case A) basis ────────────────────────────────────
        log.info("\n[1/3] Monomial (monic polynomial) basis  — no LP")
        M_A = build_collocation_matrix(n_interior, ALPHA, BETA, GAMMA, T_L, T_R)
        r_A = run_one(N, "Monomial (Case A)", M_A, b_raw, log)
        all_results.append(r_A)

        # ── 2. Monic LP (in-between) ──────────────────────────────────────
        log.info("\n[2/3] Monic LP (in-between) basis")
        mt, st, (M_monic, _, _) = time_lp(
            minimize_pauli_l1_monic,
            n_interior, ALPHA, BETA, GAMMA, T_L, T_R,
            n_trials=N_TRIALS,
        )
        log.info(f"  Monic LP time: {mt:.4f} ± {st:.4f} s")
        r_monic = run_one(N, "Monic LP", M_monic, b_raw, log,
                          lp_time_mean=mt, lp_time_std=st)
        all_results.append(r_monic)

        # ── 3. Full LP (global optimum) ───────────────────────────────────
        log.info("\n[3/3] Full LP (global optimum, always M*=I_N)")
        mt_f, st_f, (M_lp, _, _) = time_lp(
            minimize_pauli_l1,
            n_interior, ALPHA, BETA, GAMMA, T_L, T_R,
            n_trials=N_TRIALS,
        )
        log.info(f"  Full LP time: {mt_f:.4f} ± {st_f:.4f} s")
        r_lp = run_one(N, "Full LP", M_lp, b_raw, log,
                       lp_time_mean=mt_f, lp_time_std=st_f)
        all_results.append(r_lp)

        # ── Plot for this N ───────────────────────────────────────────────
        make_plot(N, [r_A, r_monic, r_lp])
        log.info(f"\nFinished N={N}\n")

    # ── Combined summary ──────────────────────────────────────────────────
    summary_path = os.path.join(LOGS_DIR, "scaling_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Scaling Experiment — ODE: u''+u'+u = t²+3t+3 on [-1,1]\n")
        f.write(f"VQLS steps={VQLS_STEPS}, trials={N_TRIALS}, seeds={VQLS_SEEDS}\n")
        f.write(f"VQLS skip threshold (HT/step): {VQLS_SKIP_THRESHOLD:,}\n")
        print_summary(all_results, stream=f)

    print(f"\nSummary saved → {summary_path}")
    print(f"Plots saved  → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
