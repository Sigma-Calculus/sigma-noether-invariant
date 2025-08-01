#!/usr/bin/env python3
# ======================================================================
#  Gauge‑Reduced σ‑Noether‑Invariant • Reproducibility Script (v2.1)
#  --------------------------------------------------------------------
#  * sparse incidence matrix (SciPy CSR)
#  * conjugate‑gradient harmonisation  ->  O(E · N_CG)
#  * supports open or periodic boundary
#  * 180° or  90° rotation (auto‑choice)
# ======================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

rng = np.random.default_rng()

# ----------------------------------------------------------------------
# 1. Build incidence matrix  d : C^0 -> C^1   for an m×n lattice
# ----------------------------------------------------------------------
def build_incidence_matrix(m: int, n: int, boundary: str = "open"):
    """
    Returns
      d  : CSR (E × V)
      edge_list : list[(u,v)]   orientation u -> v
    """
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")

    V = m * n
    edges = []

    def v(i, j):
        return (i % m) * n + (j % n)

    # horizontal edges
    for i in range(m):
        for j in range(n - (boundary == "open")):
            edges.append((v(i, j), v(i, j + 1)))

    # vertical edges
    for i in range(m - (boundary == "open")):
        for j in range(n):
            edges.append((v(i, j), v(i + 1, j)))

    E = len(edges)
    rows, cols, data = [], [], []
    for idx, (u, w) in enumerate(edges):
        rows.extend([idx, idx])
        cols.extend([u,   w  ])
        data.extend([-1.0, +1.0])

    d = sp.csr_matrix((data, (rows, cols)), shape=(E, V))
    return d, edges


# ----------------------------------------------------------------------
# 2. Signed edge‑rotation operator  S  and its order p
# ----------------------------------------------------------------------
def rotation_edge_operator(edges, m: int, n: int,
                           boundary: str = "open", rotation="auto"):
    """
    Create signed permutation S (E×E) that implements the chosen rotation.
    Returns  S (CSR)  and its order p.
    """

    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")

    E = len(edges)
    data, rows, cols = [], [], []

    def vert_idx(i, j):
        return (i % m) * n + (j % n)

    # 180° and 90° rotations in index space
    rot_180 = lambda i, j: (m - 1 - i, n - 1 - j)
    rot_90  = lambda i, j: (j, m - 1 - i)

    if rotation == "auto":
        rot = rot_90 if (boundary == "periodic" and m == n) else rot_180
    elif rotation == "180":
        rot = rot_180
    elif rotation == "90":
        if m != n:
            raise ValueError("90° rotation requires m == n")
        rot = rot_90
    else:
        raise ValueError("rotation must be 'auto', '180', or '90'")

    # edge‑dictionary for O(1) lookup
    edict = {(u, v): idx for idx, (u, v) in enumerate(edges)}

    for idx, (u, v) in enumerate(edges):
        iu, ju = divmod(u, n)
        iv, jv = divmod(v, n)

        u_rot = vert_idx(*rot(iu, ju))
        v_rot = vert_idx(*rot(iv, jv))

        if (u_rot, v_rot) in edict:
            jdx = edict[(u_rot, v_rot)]
            sign = +1.0
        elif (v_rot, u_rot) in edict:
            jdx = edict[(v_rot, u_rot)]
            sign = -1.0
        else:
            raise RuntimeError("Rotated edge not found – increase grid size")

        rows.append(jdx); cols.append(idx); data.append(sign)

    S = sp.csr_matrix((data, (rows, cols)), shape=(E, E))

    # determine permutation order ≤ 4 (90° or 180°)
    cur = sp.identity(E, format="csr")
    for k in range(1, 10):
        cur = S @ cur
        if (cur - sp.identity(E)).nnz == 0:
            return S, k
    raise RuntimeError("Permutation order > 9 – unexpected grid symmetry")


# ----------------------------------------------------------------------
# 3. Harmonic projection   σ_harm = σ + dλ    with CG solver
# ----------------------------------------------------------------------
def harmonize(d: sp.csr_matrix, sigma: np.ndarray,
              cg_tol=1e-12, cg_maxit=None):
    dT = d.transpose()
    L  = dT @ d                      # combinatorial Laplacian on 0‑cochains
    b  = -dT @ sigma

    # fix gauge λ_0 = 0  →  eliminate first row/col
    mask = np.ones(L.shape[0], dtype=bool); mask[0] = False
    Lred = L[mask][:, mask].tocsr()
    bred = b[mask]

    lam, info = spla.cg(Lred, bred, rtol=cg_tol, maxiter=cg_maxit)
    if info != 0:
        raise RuntimeError("CG did not converge (info = %d)" % info)

    lam = np.insert(lam, 0, 0.0)
    return sigma + d @ lam


# ----------------------------------------------------------------------
# 4. Q‑invariant constancy test
# ----------------------------------------------------------------------
def is_conserved(Q, rtol=1e-12, atol=1e-10):
    """combined relative / absolute tolerance"""
    return np.max(np.abs(np.diff(Q + Q[:1]))) < \
           max(atol, rtol * np.max(np.abs(Q)))


# ----------------------------------------------------------------------
# 5. Main test routine
# ----------------------------------------------------------------------
def run_test(m=8, n=8, boundary="open", k_trials=5,
             rotation="auto", cg_tol=1e-12):
    d, edges = build_incidence_matrix(m, n, boundary)
    Delta1   = d @ d.transpose()
    S, p     = rotation_edge_operator(edges, m, n, boundary, rotation)

    hodge_iso = (S @ Delta1 - Delta1 @ S).nnz == 0

    print(f"\nGrid {m}×{n},  boundary={boundary}")
    print(f"Rotation: {rotation}  → order p = {p}")
    print("Hodge‑isometric?   ", hodge_iso)

    pass_all = hodge_iso
    for t in range(k_trials):
        sigma0 = rng.integers(-3, 4, size=len(edges)).astype(float)
        seq = [harmonize(d, sigma0, cg_tol)]

        for _ in range(1, p):
            seq.append(harmonize(d, S @ seq[-1], cg_tol))

        Q = [np.dot(seq[k], seq[k] - seq[(k + 1) % p]) for k in range(p)]
        conserved = is_conserved(Q)
        pass_all &= conserved
        print(f"  Trial {t+1:2d}:  Q = {Q}   conserved? {conserved}")

    print("===> OVERALL:", "PASS" if pass_all else "FAIL")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 8×8 open grid, 180° rotation (auto)
    run_test(m=8, n=8, boundary="open", k_trials=5, rotation="auto")

    # 8×8 torus, 90° rotation
    run_test(m=8, n=8, boundary="periodic", k_trials=5, rotation="auto")
