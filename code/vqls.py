"""
VQLS comparison: Case A (monomial basis) vs LP-optimized (M* = I_N)
for Experiments 1, 2, 3.

Cost function overhead scales as O(L^2 * n_qubits) Hadamard tests per step,
where L = number of nonzero Pauli strings.  The LP preprocessing reduces L
from N^2 to 1, collapsing that cost to O(n_qubits).
"""

import sys
import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

sys.path.insert(0, "d:/Courses/math 499/code")
from collocation import (
    build_collocation_matrix,
    minimize_pauli_l1,
    count_pauli_strings,
    equidistant_interior_points,
    pauli_decompose,
)

# ---------------------------------------------------------------------------
# Pauli decomposition: real coefficients only (odd-Y terms are imaginary for
# real matrices and handled separately in the cost function)
# ---------------------------------------------------------------------------

def get_pauli_terms(M, tol=1e-10):
    """Return (coefficients, labels) of nonzero Pauli terms of M.

    Uses the scalar convention from collocation.py:
      even-Y → Re(c_P),  odd-Y → Im(c_P)
    """
    from collocation import _count_y
    raw = pauli_decompose(M, tol)
    coeffs, labels = [], []
    for label, v in raw.items():
        scalar = v.imag if (_count_y(label) % 2 == 1) else v.real
        if abs(scalar) > tol:
            coeffs.append(float(scalar))
            labels.append(label)
    return np.array(coeffs), labels


# ---------------------------------------------------------------------------
# Generic controlled-Pauli gate (for any Pauli string, applied to system
# qubits 0..n_qubits-1 controlled on ancilla_idx)
# ---------------------------------------------------------------------------

def CA(label, ancilla_idx):
    """Apply controlled-P for Pauli string `label`."""
    for position, letter in enumerate(label):
        if letter == 'X':
            qml.CNOT(wires=[ancilla_idx, position])
        elif letter == 'Y':
            qml.CY(wires=[ancilla_idx, position])
        elif letter == 'Z':
            qml.CZ(wires=[ancilla_idx, position])
        # 'I' → no gate


# ---------------------------------------------------------------------------
# VQLS core
# ---------------------------------------------------------------------------

def run_vqls(M, b_vec, steps=200, eta=0.8, q_delta=0.001, rng_seed=0, verbose=True):
    """Run VQLS for system M x = b_vec.

    M must be normalized: M_norm = M / ||M||_F.
    b_vec must be normalized: ||b_vec|| = 1.

    Returns
    -------
    cost_history : list of floats
    elapsed      : wall-clock seconds
    final_weights: np.ndarray
    """
    coeffs, labels = get_pauli_terms(M)
    n_terms = len(coeffs)
    n_qubits = int(round(np.log2(M.shape[0])))
    ancilla_idx = n_qubits
    tot_qubits = n_qubits + 1

    if verbose:
        print(f"  n_qubits={n_qubits}, Pauli terms L={n_terms}, "
              f"Hadamard tests/step~{n_terms**2 * (n_qubits + 1)}")

    b_norm = pnp.array(b_vec / np.linalg.norm(b_vec), requires_grad=False)

    dev_mu = qml.device("lightning.qubit", wires=tot_qubits)

    def variational_block(weights):
        for idx in range(n_qubits):
            qml.Hadamard(wires=idx)
        for idx, w in enumerate(weights):
            qml.RY(w, wires=idx)

    @qml.qnode(dev_mu, interface="autograd")
    def hadamard_test(weights, l, lp, j, part):
        qml.Hadamard(wires=ancilla_idx)
        if part == "Im":
            qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)
        variational_block(weights)
        CA(labels[l], ancilla_idx)
        qml.adjoint(qml.MottonenStatePreparation)(b_norm, wires=range(n_qubits))
        if j != -1:
            qml.CZ(wires=[ancilla_idx, j])
        qml.MottonenStatePreparation(b_norm, wires=range(n_qubits))
        CA(labels[lp], ancilla_idx)
        qml.Hadamard(wires=ancilla_idx)
        return qml.expval(qml.PauliZ(wires=ancilla_idx))

    def mu(weights, l, lp, j):
        re = hadamard_test(weights, l, lp, j, "Re")
        im = hadamard_test(weights, l, lp, j, "Im")
        return re + 1j * im

    def psi_norm(weights):
        norm = 0.0
        for l in range(n_terms):
            for lp in range(n_terms):
                norm += coeffs[l] * np.conj(coeffs[lp]) * mu(weights, l, lp, -1)
        return abs(norm)

    def cost_loc(weights):
        mu_sum = 0.0
        for l in range(n_terms):
            for lp in range(n_terms):
                for j in range(n_qubits):
                    mu_sum += coeffs[l] * np.conj(coeffs[lp]) * mu(weights, l, lp, j)
        mu_sum = abs(mu_sum)
        return 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))

    pnp.random.seed(rng_seed)
    w = q_delta * pnp.random.randn(n_qubits, requires_grad=True)
    opt = qml.GradientDescentOptimizer(eta)

    cost_history = []
    t0 = time.time()
    for it in range(steps):
        w, cost = opt.step_and_cost(cost_loc, w)
        cost_history.append(float(cost))
        if verbose and (it % 20 == 0 or it == steps - 1):
            print(f"    step {it:3d}  cost={cost:.6f}  t={time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    return cost_history, elapsed, w


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_experiment(name, n_interior, alpha, beta, gamma, t_L, t_R,
                   u_L, u_R, f_rhs, steps_A=20, steps_lp=150, skip_case_A=False):
    """Run VQLS comparison for Case A vs LP for one experiment.

    Case A runs for only `steps_A` steps (it's expensive); LP runs for `steps_lp`.
    Set skip_case_A=True when Case A is too costly to run (e.g. N=8).
    """
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")

    N = n_interior + 2
    n_qubits = int(round(np.log2(N)))

    col_pts = equidistant_interior_points(n_interior, t_L, t_R)
    b_raw = np.array([u_L, u_R] + [f_rhs(t) for t in col_pts], dtype=float)
    b_norm = b_raw / np.linalg.norm(b_raw)

    # --- Case A ---
    M_A = build_collocation_matrix(n_interior, alpha, beta, gamma, t_L, t_R)
    n_p_A, _ = count_pauli_strings(M_A)
    M_A_norm = M_A / np.linalg.norm(M_A)
    ht_A = n_p_A**2 * (n_qubits + 1)   # Hadamard tests per step

    if skip_case_A:
        print(f"\n[Case A] Pauli strings: {n_p_A}/{4**n_qubits}, "
              f"~{ht_A} Hadamard tests/step — skipped (too costly)")
        hist_A, t_A = [], None
    else:
        print(f"\n[Case A] Pauli strings: {n_p_A}/{4**n_qubits}, "
              f"~{ht_A} Hadamard tests/step  ({steps_A} steps)")
        hist_A, t_A, _ = run_vqls(M_A_norm, b_norm, steps=steps_A, verbose=True)

    # --- LP optimum ---
    M_lp, B_lp, _ = minimize_pauli_l1(n_interior, alpha, beta, gamma, t_L, t_R)
    n_p_lp, _ = count_pauli_strings(M_lp)
    M_lp_norm = M_lp / np.linalg.norm(M_lp)
    ht_lp = n_p_lp**2 * (n_qubits + 1)

    print(f"\n[LP opt] Pauli strings: {n_p_lp}/{4**n_qubits}, "
          f"~{ht_lp} Hadamard tests/step  ({steps_lp} steps)")
    hist_lp, t_lp, _ = run_vqls(M_lp_norm, b_norm, steps=steps_lp, verbose=True)

    # --- Summary ---
    print(f"\n--- Summary: {name} ---")
    print(f"{'Metric':<38} {'Case A':>14} {'LP opt':>14}")
    print("-" * 67)
    print(f"{'Pauli strings L':<38} {n_p_A:>14} {n_p_lp:>14}")
    print(f"{'Hadamard tests/step':<38} {ht_A:>14} {ht_lp:>14}")
    print(f"{'Speedup in tests/step':<38} {'—':>14} {f'{ht_A/max(ht_lp,1):.0f}x':>14}")
    if hist_A:
        t_A_per_step = t_A / len(hist_A)
        print(f"{'Observed s/step (Case A)':<38} {t_A_per_step:>14.2f} {'—':>14}")
        print(f"{'Observed s/step (LP opt)':<38} {'—':>14} {t_lp/steps_lp:>14.4f}")
        label_A = f'Final cost (Case A, {steps_A} steps)'
        print(f"{label_A:<38} {hist_A[-1]:>14.6f} {'—':>14}")
    label_lp = f'Final cost (LP opt, {steps_lp} steps)'
    print(f"{label_lp:<38} {'—':>14} {hist_lp[-1]:>14.6f}")
    print(f"{'Steps to cost < 0.05 (LP)':<38} {'—':>14} {_steps_to(hist_lp, 0.05):>14}")
    print(f"{'Steps to cost < 0.01 (LP)':<38} {'—':>14} {_steps_to(hist_lp, 0.01):>14}")

    return {
        'name': name, 'n_qubits': n_qubits,
        'n_paulis_A': n_p_A, 'n_paulis_lp': n_p_lp,
        'ht_A': ht_A, 'ht_lp': ht_lp,
        'hist_A': hist_A, 'hist_lp': hist_lp,
        'time_A': t_A, 'time_lp': t_lp, 'steps_lp': steps_lp,
    }


def _steps_to(history, threshold):
    for i, c in enumerate(history):
        if c < threshold:
            return i
    return ">steps"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results = []

    # Experiment 1: u'' = 0, [0,1], u(0)=0, u(1)=1
    r1 = run_experiment(
        name="Exp 1 — u''=0 on [0,1], N=4",
        n_interior=2, alpha=1, beta=0, gamma=0,
        t_L=0, t_R=1, u_L=0, u_R=1,
        f_rhs=lambda t: 0.0,
        steps_A=20, steps_lp=150,
    )
    results.append(r1)

    # Experiment 2: u''+u'+u = f, [-1,1], u(-1)=0, u(1)=2
    r2 = run_experiment(
        name="Exp 2 — u''+u'+u=f on [-1,1], N=4",
        n_interior=2, alpha=1, beta=1, gamma=1,
        t_L=-1, t_R=1, u_L=0, u_R=2,
        f_rhs=lambda t: t**2 + 3*t + 3,
        steps_A=20, steps_lp=150,
    )
    results.append(r2)

    # Experiment 3: same ODE, N=8 — Case A has 64 Pauli strings, ~16k tests/step
    # Estimated ~240s/step on CPU: skip Case A, run LP only
    r3 = run_experiment(
        name="Exp 3 — u''+u'+u=f on [-1,1], N=8",
        n_interior=6, alpha=1, beta=1, gamma=1,
        t_L=-1, t_R=1, u_L=0, u_R=2,
        f_rhs=lambda t: t**2 + 3*t + 3,
        steps_A=20, steps_lp=150,
        skip_case_A=True,
    )
    results.append(r3)

    # Final comparison table
    print(f"\n{'='*70}")
    print("  OVERALL COMPARISON")
    print(f"{'='*70}")
    header = f"{'Experiment':<34} {'L_A':>5} {'L_LP':>5} {'HT_A/step':>10} {'HT_LP/step':>10} {'HT ratio':>9}"
    print(header)
    print("-" * 70)
    for r in results:
        ratio = r['ht_A'] / max(r['ht_lp'], 1)
        print(f"{r['name']:<34} {r['n_paulis_A']:>5} {r['n_paulis_lp']:>5} "
              f"{r['ht_A']:>10} {r['ht_lp']:>10} {ratio:>9.0f}x")
