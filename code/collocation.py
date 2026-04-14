import numpy as np
import sympy as sp
from fractions import Fraction
from itertools import product as iproduct

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
    """Return (count, nonzero_coeffs_dict) of nonzero Pauli coefficients."""
    coeffs = pauli_decompose(M, tol)
    nonzero = {k: v for k, v in coeffs.items() if abs(v) > tol}
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

if __name__ == '__main__':
    experiment1()
    print()
    experiment2()
