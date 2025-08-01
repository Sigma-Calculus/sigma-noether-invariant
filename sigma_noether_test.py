"""
=======================================================================
Discrete Noether‑Invariant Test Suite  (signed edges, v2)
=======================================================================
"""

import numpy as np
rng = np.random.default_rng()

# --------------------------------------------------------------------
# 1. Inzidenzmatrix d  für ein m×n‑Gitter
# --------------------------------------------------------------------
def build_incidence_matrix(m: int, n: int, boundary: str = "open"):
    V = m * n
    edge_list = []

    def v(i, j):
        return (i % m) * n + (j % n)

    # horizontal
    for i in range(m):
        rng_j = range(n - (boundary == "open"))
        for j in rng_j:
            edge_list.append((v(i, j), v(i, j + 1)))

    # vertical
    rng_i = range(m - (boundary == "open"))
    for i in rng_i:
        for j in range(n):
            edge_list.append((v(i, j), v(i + 1, j)))

    E = len(edge_list)
    d = np.zeros((E, V))
    for idx, (u, w) in enumerate(edge_list):
        d[idx, u] = -1
        d[idx, w] = +1
    return d, edge_list


# --------------------------------------------------------------------
# 2. Erzeuge signierten Kanten‑Operator S und Vertex‑Permutation V
# --------------------------------------------------------------------
def rotation_edge_operator(edge_list, m: int, n: int, boundary: str ="open",
                           rotation="auto"):
    """
    rotation = "auto" | "180" | "90"
     "auto": 180° für open, 90° für quadratischen Torus
     "180":  immer 180°
     "90":   nur zulässig, wenn m==n  (prüfen!)
    
      open:     S ist reine Permutation   (± = +1)
      periodic: S ist signierte Permutation (±1)
      
    Gibt   S (E×E)  und Ordnung p zurück
    """

    E = len(edge_list)
    S = np.zeros((E, E)) # signed orthogonal matrix

    # Hilfsfunktionen 
    def rot_180(i, j):
        return (m - 1 - i, n - 1 - j)

    def rot_90(i, j):
        return (j, m - 1 - i)

    def vert_id(coord):
        return (coord[0] % m) * n + (coord[1] % n)
    
    # --------------------------------
    #  Auswahl der Rotationsfunktion
    # --------------------------------
    roter = {
        "180": lambda i, j: rot_180(i, j),
        "90":  lambda i, j: rot_90(i, j),
        "auto": (rot_90 if boundary == "closed" and m == n else rot_180)
    }
    
    if rotation not in roter:
        raise ValueError("rotation must be 'auto', '180', or '90'")

    rot = roter[rotation]

    # Lookup (u,v) → idx
    edge_dict = {(u, v): idx for idx, (u, v) in enumerate(edge_list)}

    for idx, (u, v) in enumerate(edge_list):
        iu, ju = divmod(u, n)
        iv, jv = divmod(v, n)

        if boundary == "open":
            ru, rv = rot(iu, ju), rot(iv, jv)
        else: # boundary is periodic.
            ru, rv = rot(iu, ju), rot(iv, jv)

        u_rot = vert_id(ru)
        v_rot = vert_id(rv)

        # gleiche Orientierung?
        if (u_rot, v_rot) in edge_dict:
            jdx = edge_dict[(u_rot, v_rot)]
            S[jdx, idx] = +1
        elif (v_rot, u_rot) in edge_dict:
            jdx = edge_dict[(v_rot, u_rot)]
            S[jdx, idx] = -1
        else:
            raise RuntimeError("Rotated edge not found – grid too small?")
    # Ordnung ermitteln
    cur = np.identity(E)
    for k in range(1, 10):
        cur = S @ cur
        if np.allclose(cur, np.identity(E)):
            return S, k
    raise RuntimeError("Permutation order > 9 ?!")
    

# ----------------------------
# 3. Harmonischer Projektor
# ----------------------------
def harmonize(d, dT, sigma):
    L = dT @ d
    lam = np.linalg.solve(L[1:, 1:], (-dT @ sigma)[1:])
    lam = np.concatenate(([0.], lam))
    return sigma + d @ lam

# ------------------------------------------
# Konstanztest mit kombinierten Toleranzen
# ------------------------------------------
def is_conserved(Q, rtol=1e-12, atol=1e-10):
    return np.max(np.abs(np.diff(Q + Q[:1]))) < max(atol, rtol * np.max(np.abs(Q)))


# ----------------------
# 4. Haupt‑Testfunktion
# ----------------------
def run_test(m=4, n=4, boundary="open", k_trials=5, rotation="auto"):
    d, edge_list = build_incidence_matrix(m, n, boundary)
    dT      = d.T
    Delta1  = d @ dT

    S, p = rotation_edge_operator(edge_list, m, n, boundary, rotation)
    
    hodge_iso = np.allclose(S @ Delta1, Delta1 @ S)

    print(f"\nGrid {m}×{n},  boundary={boundary}")
    print(f"Permutation order p = {p}")
    print("Hodge‑isometric?    ", hodge_iso)

    pass_all = hodge_iso
    for t in range(k_trials):
        sigma0     = rng.integers(-3, 4, size=len(edge_list)).astype(float)
        s          = harmonize(d, dT, sigma0)
        seq        = [s]

        # RG‑Schritte + Harmonisierung
        for _ in range(1, p):
            s = harmonize(d, dT, S @ s)
            seq.append(s)

        Q = [np.dot(seq[k], seq[k] - seq[(k + 1) % p]) for k in range(p)]
        conserved = is_conserved(Q)
        pass_all &= conserved

        print(f"  Trial {t+1}:  Q = {Q}   conserved? {conserved}")

    print("===> OVERALL:", "PASS" if pass_all else "FAIL")


if __name__ == "__main__":
    # 180°‑Rotation, offenes 4×4‑Gitter
    run_test(m=8, n=8, boundary="open", k_trials=10, rotation="90")

    # 90°‑Rotation, toroidales 4×4‑Gitter
    run_test(m=8, n=8, boundary="periodic", k_trials=10, rotation="180")
