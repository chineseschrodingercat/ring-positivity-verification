"""Certificate that the Hildebrand decic is reduced of degree ten along a rational line.

Restricts the decic stored in decic.json (identity cyclic order, 313 terms) to the
line A + xB of line_certificate.json, compares the result with the recorded integer
polynomial up to a nonzero constant, and checks that it has degree ten, a leading
coefficient prime to p = 17, and is irreducible modulo p.  Then it is irreducible
over Q, hence square-free: the line meets the component in ten distinct points.
Run: python line_certificate.py
"""
import json
from pathlib import Path
import sympy as sp

ROOT = Path(__file__).resolve().parent
decic = json.loads((ROOT / "decic.json").read_text())
cert = json.loads((ROOT / "line_certificate.json").read_text())
A, B = cert["line_A"], cert["line_B"]
x = sp.symbols("x")
values = {}
for name in decic["variables"]:
    i, j = int(name[1]), int(name[2])
    values[name] = A[i][j] + x * B[i][j]
poly = sp.Integer(0)
for coef, exps in decic["terms"]:
    term = sp.Integer(coef)
    for name, e in zip(decic["variables"], exps):
        if e:
            term *= values[name] ** e
    poly += term
h = sp.Poly(sp.expand(poly), x)
recorded = sp.Poly(cert["restriction_coeffs_high_to_low"], x)
ratio = sp.cancel(h.as_expr() / recorded.as_expr())
assert ratio.is_number and ratio != 0, "restriction differs from the recorded polynomial"
p = cert["prime"]
assert h.degree() == 10 and recorded.LC() % p != 0
assert sp.Poly(recorded.as_expr(), x, modulus=p).is_irreducible
assert sp.gcd(recorded, recorded.diff(x)).degree() == 0
print(f"degree {h.degree()}, ratio to recorded polynomial {ratio}, irreducible mod {p}: ten distinct intersection points")
