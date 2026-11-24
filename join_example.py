"""Exact check of the order-eight example of a plane-plane join (Example in Section 9).

For a canonical normal Y(1, theta) and a word in E, R, M, the linear span of the
normal fiber at a generic point of the exposed face is cut out by the first-order
conditions (Y' v)_i = 0 for every zero vector v of every window and every i in the
support of v (an interior zero forces Y' v = 0).  The word gives a hypersurface
exactly when this space is the line spanned by Y itself (Lemma on normal fibers).

Angles are given by rational half-angle tangents, so all entries are rational.
Run: python join_example.py
"""
from fractions import Fraction as Fr
import sympy as sp

PI = "pi"


def trig(u):
    """cos and sin of theta = 2 arctan(u); u = PI means theta = pi."""
    if u == PI:
        return Fr(-1), Fr(0)
    u = Fr(u)
    return (1 - u*u) / (1 + u*u), 2*u / (1 + u*u)


def canonical_normal(n, us):
    cs = [trig(u) for u in us]
    Y = sp.zeros(n, n)
    for i in range(n):
        Y[i, i] = 1
        c, s = cs[i]
        Y[i, (i+1) % n] = Y[(i+1) % n, i] = -c
        c1, s1 = cs[(i+1) % n]
        csum = c*c1 - s*s1
        ssum = s*c1 + c*s1
        clipped = (us[i] == PI or us[(i+1) % n] == PI or ssum < 0 or (ssum == 0 and csum < 0))
        Y[i, (i+2) % n] = Y[(i+2) % n, i] = -1 if clipped else csum
    return Y


def zero_vectors(us, j, n):
    a, b = us[j], us[(j+1) % n]
    ca, sa = trig(a); cb, sb = trig(b)
    out = []
    if a != PI and ca == 1:
        out.append((1, 1, 0))
    if b != PI and cb == 1:
        out.append((0, 1, 1))
    ssum = sa*cb + ca*sb; csum = ca*cb - sa*sb
    sat = (a == PI or b == PI or ssum < 0 or (ssum == 0 and csum < 0))
    if sat:
        out.append((1, 0, 1))
    elif sa > 0 and sb > 0:
        out.append((sb, ssum, sa))
    return out


def fiber_dimension(n, us):
    Y = canonical_normal(n, us)
    entries = [(i, i) for i in range(n)] + [(i, (i+1) % n) for i in range(n)] + [(i, (i+2) % n) for i in range(n)]
    var = sp.symbols(f"y0:{len(entries)}")
    Z = sp.zeros(n, n)
    for x, (i, j) in zip(var, entries):
        Z[i, j] = Z[j, i] = x
    eqs = []
    for j in range(n):
        T = [j, (j+1) % n, (j+2) % n]
        vs = zero_vectors(us, j, n)
        if len(vs) >= 2:                      # R window: an interior zero exists
            vs = vs + [tuple(sum(v[k] for v in vs) for k in range(3))]
        for v in vs:
            for a in range(3):
                if v[a] != 0:
                    eqs.append(sum(Z[T[a], T[b]] * v[b] for b in range(3)))
    M = sp.Matrix([[sp.diff(e, x) for x in var] for e in eqs])
    base = sp.Matrix([Y[i, j] for (i, j) in entries])
    assert M * base == sp.zeros(M.rows, 1), "the base normal must satisfy the conditions"
    return len(var) - M.rank()


if __name__ == "__main__":
    q, w = Fr(1, 4), Fr(3, 2)        # tan(theta/2) for small and for large interior angles
    # MEMEEMRR: theta_0..theta_7 = pi, s, L, L, s, s, pi, 0
    good = [PI, q, w, w, q, q, PI, 0]
    # MEMEMRRR: theta_0..theta_7 = pi, s, L, L, s, pi, 0, 0  (both joined blocks of length one)
    bad = [PI, q, w, w, q, PI, 0, 0]
    for name, us in (("MEMEEMRR", good), ("MEMEMRRR", bad)):
        print(name, "normal-fiber dimension:", fiber_dimension(8, us))
    # closed-form determinant of the join system for the good word
    th = {k: 2*sp.atan(sp.Rational(v)) for k, v in ((1, q), (2, w), (3, w), (4, q), (5, q))}
    det = sp.sin(th[2])*sp.sin(th[1]+th[2])*sp.sin(th[4])*sp.sin(th[4]+th[5])
    print("join determinant sin(t2)sin(t1+t2)sin(t4)sin(t4+t5) =", sp.nsimplify(sp.simplify(sp.expand_trig(det))))
