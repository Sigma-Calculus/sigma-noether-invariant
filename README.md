

# σ‑Noether‑Invariant – minimal reproduction script

This repository accompanies the paper

> O. Sievers, *Gauge‑Reduced Discrete Noether Invariants
> in the Sigma Calculus* (2025).
>
> https://zenodo.org/records/16669372

It provides a **single self‑contained Python script**
that reproduces all numerical checks.

## Files
| File | Description |
|------|-------------|
| `sigma_noether_test.py` | Builds a lattice, performs harmonic projection, applies a symmetry operator, and verifies δQ ≤ 1e‑10. |
| `requirements.txt` | Only `numpy >= 1.24` is required. |

## Usage
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python sigma_noether_test.py    # default: 8×8 open grid
````

The script prints

```
Grid 8×8,  boundary=open
Permutation order p = 4
Hodge‑isometric?     True
  Trial 1:  Q = [149.1933694765365, 149.19336947653647, 149.19336947653647, 149.19336947653647]   conserved? True
  ...
===> OVERALL: PASS

Grid 8×8,  boundary=periodic
Permutation order p = 2
Hodge‑isometric?     True
  Trial 1:  Q = [223.08552170868347, 223.0855217086835]   conserved? True
  ...
===> OVERALL: PASS
```

which confirms the exact conservation law
`widehat Q_{k+1} = widehat Q_k` within numerical precision.

## License

MIT for code, CC‑BY 4.0 for documentation.
