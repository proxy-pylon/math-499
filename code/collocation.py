import numpy as np
import sympy as sp
from fractions import Fraction
from itertools import product as iproduct
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Pauli decomposition utilities
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = {'I': _I, 'X': _X, 'Y': _Y, 'Z': _Z}


def pauli_string_matrix(label: str) -> np.ndarray:
    """Tensor product for a Pauli label like 'IXZ'."""
    result = np.array([[1.0 + 0j]])
    for c in label:
        result = np.kron(result, _PAULIS[c])
    return result


def _count_y(label: str) -> int:
    return label.count('Y')


def pauli_decompose(M: np.ndarray, tol: float = 1e-10) -> dict:
    """Decompose M into Pauli strings.  Returns {label: coefficient}.

    For n qubits (2^n x 2^n matrix):
        c_P = Tr(P @ M) / 2^n
    """
    n = M.shape[0]
    n_qubits = int(round(np.log2(n)))
    assert 2**n_qubits == n, "Matrix size must be a power of 2"

    coeffs = {}
    for labels in iproduct('IXYZ', repeat=n_qubits):
        label = ''.join(labels)
        P = pauli_string_matrix(label)
        c = np.trace(P @ M) / n
        coeffs[label] = c
    return coeffs


def count_pauli_strings(M: np.ndarray, tol: float = 1e-10):
    """Return (count, nonzero_coeffs_dict) of nonzero Pauli coefficients.

    For each Pauli string the relevant scalar is:
      - even-Y strings: Re(c_P)  (real coefficient for real M)
      - odd-Y  strings: Im(c_P)  (imaginary coefficient for real M)
    """
    coeffs = pauli_decompose(M, tol)
    nonzero = {}
    for label, v in coeffs.items():
        scalar = v.imag if (_count_y(label) % 2 == 1) else v.real
        if abs(scalar) > tol:
            nonzero[label] = scalar
    return len(nonzero), nonzero


# ---------------------------------------------------------------------------
# Core collocation matrix builder
# ---------------------------------------------------------------------------

def equidistant_interior_points(n: int, t_L=-1, t_R=1):
    """n equidistant interior collocation points on (t_L, t_R)."""
    h = (t_R - t_L) / (n + 1)
    return [t_L + (k + 1) * h for k in range(n)]


def poly_eval(coeffs, t):
    """Evaluate polynomial sum_k coeffs[k]*t^k at t."""
    return sum(c * t**k for k, c in enumerate(coeffs))


def apply_L_to_poly(coeffs, alpha, beta, gamma):
    """Apply L = alpha*D^2 + beta*D + gamma to a polynomial.

    coeffs: list [a0, a1, ..., an] for a0 + a1*t + ... + an*t^n
    Returns the coefficient list of L[poly].
    """
    n = len(coeffs)
    result = [sp.Integer(0)] * n

    for k, c in enumerate(coeffs):
        result[k] += gamma * c
        if k >= 1:
            result[k - 1] += beta * k * c
        if k >= 2:
            result[k - 2] += alpha * k * (k - 1) * c

    return result


def build_collocation_matrix(n_interior: int, alpha, beta, gamma,
                              t_L=-1, t_R=1, C=None, use_sympy=False):
    """Build the collocation matrix for L[u] = alpha*u'' + beta*u' + gamma*u.

    Parameters
    ----------
    n_interior : int
        Number of interior collocation points (also = degree of freedom beyond BCs).
    alpha, beta, gamma : scalar or sympy expression
        Coefficients of the differential operator.
    t_L, t_R : float
        Domain endpoints.
    C : 2-D array-like, shape (N, N) where N = n_interior + 2, optional
        Basis coefficient matrix (lower-triangular): C[j][k] = coefficient of t^k in phi_j.
        Defaults to the identity (monomial basis, Case A).
    use_sympy : bool
        If True, return a sympy Matrix; otherwise a numpy array (requires numeric C).

    Returns
    -------
    M : sympy.Matrix or numpy.ndarray  of shape (N, N)

    Row ordering
    ------------
    Row 0      : BC at t_L  (evaluate phi_j at t_L)
    Row 1      : BC at t_R  (evaluate phi_j at t_R)
    Rows 2..N-1: ODE at equidistant interior points (evaluate L[phi_j] at t_k)
    """
    N = n_interior + 2

    if C is None:
        C = [[1 if i == j else 0 for j in range(N)] for i in range(N)]

    col_pts = equidistant_interior_points(n_interior, t_L, t_R)

    rows = []
    rows.append([poly_eval(C[j], t_L) for j in range(N)])
    rows.append([poly_eval(C[j], t_R) for j in range(N)])

    for t_k in col_pts:
        row = []
        for j in range(N):
            Lphi = apply_L_to_poly(list(C[j]), alpha, beta, gamma)
            row.append(poly_eval(Lphi, t_k))
        rows.append(row)

    if use_sympy:
        return sp.Matrix(rows)
    return np.array(rows, dtype=float)


def make_symbolic_C(N: int, prefix='C'):
    """Create symbolic lower-triangular C matrix.  C[j][k] = 0 for k > j."""
    syms = {}
    C = []
    for j in range(N):
        row = []
        for k in range(N):
            if k > j:
                row.append(sp.Integer(0))
            else:
                sym = sp.Symbol(f'{prefix}_{j}{k}')
                syms[f'{prefix}_{j}{k}'] = sym
                row.append(sym)
        C.append(row)
    return C, syms


# ---------------------------------------------------------------------------
# Linear programming: minimize Pauli string count via L1
# ---------------------------------------------------------------------------

def _pauli_jacobian(N: int) -> tuple[np.ndarray, list[str]]:
    """Compute the real Jacobian J where J[p, i*N+j] = d(scalar_c_P)/d(M[i,j]).

    c_P = Tr(P M) / N.  dc_P/dM[i,j] = P[j,i] / N.

    For real M:
      - even-Y strings: c_P is real  → use Re(P[j,i]/N)
      - odd-Y  strings: c_P is purely imaginary → use Im(P[j,i]/N)

    Returns J (n_pauli x N^2) and pauli_labels.
    """
    n_qubits = int(round(np.log2(N)))
    assert 2**n_qubits == N
    pauli_labels = [''.join(l) for l in iproduct('IXYZ', repeat=n_qubits)]
    n_pauli = len(pauli_labels)
    J = np.zeros((n_pauli, N * N))
    for p, label in enumerate(pauli_labels):
        P = pauli_string_matrix(label)
        odd_y = (_count_y(label) % 2 == 1)
        for i in range(N):
            for j in range(N):
                val = complex(P[j, i]) / N
                J[p, i * N + j] = val.imag if odd_y else val.real
    return J, pauli_labels


def minimize_pauli_l1(n_interior: int, alpha, beta, gamma, t_L=-1, t_R=1):
    """Find the collocation matrix M with minimum L1 Pauli content.

    Basis change identity (from ideas.pdf):
        M = C_A * B^T
    where C_A is the monomial (Case A) collocation matrix and B is any invertible
    basis transformation matrix.  For ANY invertible target M we can recover a
    valid basis via B = (C_A^{-1} M)^T, so M is a completely free variable.

    LP formulation — variables: m = vec(M) (N^2 entries, row-major) and
    slack variables t_P >= 0 (one per Pauli string):

        min   sum_P t_P
        s.t.  J m - t <= 0        (t_P >= +c_P(M))
             -J m - t <= 0        (t_P >= -c_P(M))
              Tr(M)   = N          (normalization: c_II = 1, prevents M -> 0)

    All constraints are linear in m.  Tr(M) = N fixes the scale and guarantees
    the trivial solution M = 0 is excluded; the global minimum is M = I_N
    (1 Pauli string, c_II = 1, all others = 0).

    Returns
    -------
    M_opt  : np.ndarray (N x N)  optimal collocation matrix
    B_opt  : np.ndarray (N x N)  basis matrix  (phi_j = sum_k B[j,k] t^k)
    result : OptimizeResult from scipy.optimize.linprog
    """
    N = n_interior + 2
    J, pauli_labels = _pauli_jacobian(N)
    n_pauli = len(pauli_labels)

    # Objective: min sum(t)
    c_obj = np.concatenate([np.zeros(N * N), np.ones(n_pauli)])

    # Inequality: [J, -I; -J, -I] [m; t] <= 0
    A_ub = np.vstack([
        np.hstack([ J, -np.eye(n_pauli)]),
        np.hstack([-J, -np.eye(n_pauli)]),
    ])
    b_ub = np.zeros(2 * n_pauli)

    # Equality: Tr(M) = N  (m[j*N+j] are the diagonal entries)
    A_eq = np.zeros((1, N * N + n_pauli))
    for j in range(N):
        A_eq[0, j * N + j] = 1.0
    b_eq = np.array([float(N)])

    bounds = [(None, None)] * (N * N) + [(0, None)] * n_pauli

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    M_opt = result.x[:N * N].reshape(N, N)

    # Recover basis B: M = C_A B^T  =>  B^T = C_A^{-1} M
    C_A = build_collocation_matrix(n_interior, alpha, beta, gamma, t_L, t_R)
    B_opt = np.linalg.solve(C_A, M_opt).T   # B = (C_A^{-1} M)^T

    return M_opt, B_opt, result


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment1():
    """u'' = 0 on [0,1], u(0)=0, u(1)=1.  Expected solution: u(t)=t."""
    print("=" * 60)
    print("Experiment 1 — u'' = 0 on [0,1]")
    print("=" * 60)

    # Case A: monomial basis
    M = build_collocation_matrix(n_interior=2, alpha=1, beta=0, gamma=0, t_L=0, t_R=1)
    b = np.array([0, 1, 0, 0], dtype=float)

    print("\nCase A (monomial basis):")
    print("M =\n", M)
    print(f"det(M) = {np.linalg.det(M):.4f}")
    a = np.linalg.solve(M, b)
    print(f"coefficients a = {a}")
    terms = [f"{a[k]:.4g}*t^{k}" for k in range(len(a)) if abs(a[k]) > 1e-10]
    print("u(t) =", " + ".join(terms) if terms else "0")
    n_paulis, _ = count_pauli_strings(M)
    print(f"Pauli string count: {n_paulis}/16")

    # Case B: optimal 3-Pauli basis from the paper
    C_opt = [
        [2,  0,  0,  0],
        [2,  2,  0,  0],
        [0, -1,  1,  0],
        [0, -1,  0,  1],
    ]
    M_B = build_collocation_matrix(n_interior=2, alpha=1, beta=0, gamma=0, t_L=0, t_R=1, C=C_opt)

    print("\nCase B (optimal 3-Pauli basis):")
    print("M =\n", M_B)
    print(f"det(M) = {np.linalg.det(M_B):.4f}")
    n_paulis, paulis = count_pauli_strings(M_B)
    print(f"Pauli string count: {n_paulis}/16")
    print("Nonzero Paulis:", {k: round(v.real, 6) for k, v in paulis.items()})


def experiment2():
    """u'' + u' + u = t^2+3t+3 on [-1,1], u(-1)=0, u(1)=2.  Expected: u(t)=t^2+t."""
    print("=" * 60)
    print("Experiment 2 — u''+u'+u = t^2+3t+3 on [-1,1]")
    print("=" * 60)

    f = lambda t: t**2 + 3*t + 3
    col_pts = equidistant_interior_points(2, -1, 1)
    b = np.array([0, 2, f(col_pts[0]), f(col_pts[1])], dtype=float)

    # Case A: monomial basis
    M = build_collocation_matrix(n_interior=2, alpha=1, beta=1, gamma=1, t_L=-1, t_R=1)

    print("\nCase A (monomial basis):")
    fmt = {'float_kind': lambda x: f"{Fraction(x).limit_denominator(1000)!s:>8}"}
    print("M =\n", np.array2string(M, formatter=fmt))
    print(f"det(M) = {np.linalg.det(M):.4f}")
    a = np.linalg.solve(M, b)
    print(f"coefficients a = {np.round(a, 10)}  (expected [0, 1, 1, 0])")
    n_paulis, paulis = count_pauli_strings(M)
    print(f"Pauli string count: {n_paulis}/16")
    print("Nonzero Paulis:", list(paulis.keys()))

    # Case B: symbolic matrix
    print("\nCase B (symbolic generalized basis):")
    C_sym, _ = make_symbolic_C(4)
    M_sym = build_collocation_matrix(n_interior=2, alpha=1, beta=1, gamma=1,
                                     t_L=-1, t_R=1, C=C_sym, use_sympy=True)
    sp.pprint(M_sym)

def experiment_lp(n_interior=2, alpha=1, beta=0, gamma=0, t_L=0, t_R=1,
                  label="Experiment 1"):
    """Run LP Pauli minimization and report results."""
    print("=" * 60)
    print(f"LP Pauli minimization — {label}")
    print("=" * 60)

    M_opt, B_opt, result = minimize_pauli_l1(
        n_interior, alpha, beta, gamma, t_L, t_R)

    print(f"LP status: {result.message}")
    print(f"L1 objective (sum |c_P|): {result.fun:.6f}")

    n_paulis, paulis = count_pauli_strings(M_opt)
    print(f"\nOptimal M =\n{np.round(M_opt, 6)}")
    print(f"det(M) = {np.linalg.det(M_opt):.6f}")
    print(f"Pauli string count: {n_paulis}/16")
    print("Nonzero Paulis:", {k: round(float(v), 6) for k, v in paulis.items()})

    print(f"\nBasis B  (phi_j = sum_k B[j,k] t^k):")
    for j in range(B_opt.shape[0]):
        vals = [f"{v:+.4f}" for v in B_opt[j]]
        print(f"  phi_{j}: {vals}")
    print(f"det(B) = {np.linalg.det(B_opt):.6f}")


def experiment3():
    """u'' + u' + u = t^2+3t+3 on [-1,1], u(-1)=0, u(1)=2, 6 interior points (N=8, 3 qubits).

    Exact solution: u(t) = t^2 + t.
    """
    print("=" * 60)
    print("Experiment 3 — u''+u'+u = t^2+3t+3 on [-1,1], N=8 (3 qubits)")
    print("=" * 60)

    n_interior = 6   # N = 8
    alpha, beta, gamma = 1, 1, 1
    t_L, t_R = -1, 1

    f = lambda t: t**2 + 3*t + 3
    col_pts = equidistant_interior_points(n_interior, t_L, t_R)
    b = np.array([0, 2] + [f(t) for t in col_pts], dtype=float)

    # Case A: monomial basis
    M_A = build_collocation_matrix(n_interior, alpha, beta, gamma, t_L, t_R)
    print(f"\nCase A (monomial basis), N={M_A.shape[0]}:")
    print(f"det(M_A) = {np.linalg.det(M_A):.4f}")
    a_A = np.linalg.solve(M_A, b)
    print(f"Coefficients a (first 3): {np.round(a_A[:3], 8)}  (expected [0,1,1])")
    n_p, _ = count_pauli_strings(M_A)
    print(f"Pauli count: {n_p}/64")

    # LP global optimum
    print("\nRunning LP...")
    M_opt, B_opt, result = minimize_pauli_l1(n_interior, alpha, beta, gamma, t_L, t_R)
    print(f"LP status: {result.message}")
    print(f"L1 objective: {result.fun:.8f}")
    n_p_opt, paulis_opt = count_pauli_strings(M_opt)
    print(f"Pauli count: {n_p_opt}/64")
    print(f"Nonzero Paulis: {list(paulis_opt.keys())}")
    print(f"det(M_opt) = {np.linalg.det(M_opt):.8f}")
    print(f"det(B_opt) = {np.linalg.det(B_opt):.6e}")

    # Verify solution
    a_opt = np.linalg.solve(M_opt, b)
    t_test = np.array([-1, -0.5, 0, 0.5, 1])
    print("\nSolution verification:")
    for t in t_test:
        phi = np.array([sum(B_opt[j, k] * t**k for k in range(8)) for j in range(8)])
        u = float(np.dot(a_opt, phi))
        exact = t**2 + t
        print(f"  u({t:+.1f}) = {u:+.8f},  exact = {exact:+.8f}")

    print("\nBasis B_opt (phi_j = sum_k B[j,k] t^k):")
    for j in range(8):
        vals = [f"{v:+.6f}" for v in B_opt[j]]
        print(f"  phi_{j}: {vals}")


if __name__ == '__main__':
    experiment1()
    print()
    experiment2()
    print()
    experiment_lp(n_interior=2, alpha=1, beta=0, gamma=0, t_L=0, t_R=1,
                  label="Experiment 1 — u''=0 on [0,1]")
    print()
    experiment_lp(n_interior=2, alpha=1, beta=1, gamma=1, t_L=-1, t_R=1,
                  label="Experiment 2 — u''+u'+u=f on [-1,1]")
    print()
    experiment3()
