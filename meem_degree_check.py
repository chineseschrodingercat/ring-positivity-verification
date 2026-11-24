"""Replay finite-field pencil computations for the word MEEMR...R.

The computed degrees describe factors of specialized polynomial gcds.
Use --replay to check the four saved pencils. Without options, search
for pencils with the published random seed.
"""
import argparse
import json
from pathlib import Path
from random import Random


def run_for_prime(p, orders=(6, 7, 8), saved=None):
    def inv(x):
        return pow(x % p, p - 2, p)

    def det_mod(M):
        n = len(M)
        A = [row[:] for row in M]
        ans = 1
        for i in range(n):
            piv = next((r for r in range(i, n) if A[r][i] % p), None)
            if piv is None:
                return 0
            if piv != i:
                A[i], A[piv] = A[piv], A[i]
                ans = (p - ans) % p
            pv = A[i][i] % p
            ans = ans * pv % p
            iv = inv(pv)
            for r in range(i + 1, n):
                f = A[r][i] * iv % p
                if f:
                    for c in range(i, n):
                        A[r][c] = (A[r][c] - f * A[i][c]) % p
        return ans

    def p_strip(a):
        while len(a) > 1 and a[-1] % p == 0:
            a.pop()
        return [x % p for x in a] if a else [0]

    def p_mul(a, b):
        res = [0] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    res[i + j] = (res[i + j] + x * y) % p
        return p_strip(res)

    def p_divmod(a, b):
        a, b = p_strip(a[:]), p_strip(b[:])
        db = len(b) - 1
        if len(a) - 1 < db:
            return [0], a
        ib = inv(b[-1])
        q = [0] * (len(a) - db)
        while len(a) - 1 >= db and a != [0]:
            da = len(a) - 1
            coef = a[-1] * ib % p
            shift = da - db
            q[shift] = coef
            for j in range(db + 1):
                a[shift + j] = (a[shift + j] - coef * b[j]) % p
            a = p_strip(a)
        return p_strip(q), a

    def p_gcd(a, b):
        a, b = p_strip(a[:]), p_strip(b[:])
        while b != [0]:
            _, r = p_divmod(a, b)
            a, b = b, r
        if a != [0]:
            ia = inv(a[-1])
            a = [x * ia % p for x in a]
        return a

    def p_powmod(base, exp, modp):
        res = [1]
        cur = p_divmod(base, modp)[1]
        while exp > 0:
            if exp & 1:
                res = p_divmod(p_mul(res, cur), modp)[1]
            cur = p_divmod(p_mul(cur, cur), modp)[1]
            exp >>= 1
        return res

    def ddf_degrees(f):
        f = p_strip(f[:])
        ia = inv(f[-1])
        f = [x * ia % p for x in f]
        deriv = lambda g: p_strip([(i * g[i]) % p for i in range(1, len(g))]) if len(g) > 1 else [0]
        c = p_gcd(f, deriv(f))
        w, _ = p_divmod(f, c)
        i, sqf_parts, out = 1, [], []
        while w != [1]:
            y = p_gcd(w, c)
            z, _ = p_divmod(w, y)
            if len(z) > 1:
                sqf_parts.append((z, i))
            w = y
            c, _ = p_divmod(c, y)
            i += 1
        for part, m in sqf_parts:
            cur, h, d = part[:], [0, 1], 1
            while len(cur) - 1 >= 2 * d:
                h = p_powmod(h, p, cur)
                hx = h[:] + [0] * max(0, 2 - len(h))
                hx[1] = (hx[1] - 1) % p
                g_d = p_gcd(cur, p_strip(hx))
                if len(g_d) > 1:
                    for _ in range((len(g_d) - 1) // d):
                        out.append((d, m))
                    cur, _ = p_divmod(cur, g_d)
                    h = p_divmod(h, cur)[1]
                d += 1
            if len(cur) > 1:
                out.append((len(cur) - 1, m))
        return sorted(out)

    def interp(xs, ys):
        m = len(xs)
        master = [1]
        for xj in xs:
            master = p_mul(master, [(-xj) % p, 1])
        tot = [0] * m
        for i, xi in enumerate(xs):
            q, _ = p_divmod(master, [(-xi) % p, 1])
            den = 1
            for j, xj in enumerate(xs):
                if i != j:
                    den = den * (xi - xj) % p
            sc = ys[i] * inv(den) % p
            for k in range(len(q)):
                tot[k] = (tot[k] + sc * q[k]) % p
        return p_strip(tot)

    def poly_res_fixed(a, da, b, db):
        a_pad = (a + [0] * (da + 1))[:da + 1]
        b_pad = (b + [0] * (db + 1))[:db + 1]
        N = da + db
        S = [[0] * N for _ in range(N)]
        ra, rb = a_pad[::-1], b_pad[::-1]
        for r in range(db):
            for j, val in enumerate(ra):
                S[r][r + j] = val
        for r in range(da):
            for j, val in enumerate(rb):
                S[db + r][r + j] = val
        return det_mod(S)

    def get_eps(n):
        eps = [0] * n
        eps[4] = eps[5] = 1
        for k in range(5, n - 1):
            eps[k + 1] = -eps[k]
        eps[0] = eps[1] = -eps[n - 1]
        eps[2] = -eps[0]
        eps[3] = -eps[5]
        return eps

    def random_band(n, k, rng):
        A = [[0] * n for _ in range(n)]
        for i in range(n):
            A[i][i] = rng.randrange(1, p)
            for d in range(1, k + 1):
                j = (i + d) % n
                v = rng.randrange(1, p)
                A[i][j] = A[j][i] = v
        return A

    def sample_meem_point(n, rng):
        r = [rng.randrange(5, p - 5) for _ in range(n)]
        u1, u2, u3 = [rng.randrange(5, p - 5) for _ in range(3)]
        cs = lambda u: (((1 - u * u) * inv(1 + u * u)) % p, (2 * u * inv(1 + u * u)) % p)
        c1, s1 = cs(u1)
        c2, s2 = cs(u2)
        c3, s3 = cs(u3)
        Am = [[0] * n for _ in range(n)]
        def add_outer(idx, vec, wt):
            for a in range(len(idx)):
                for b in range(len(idx)):
                    Am[idx[a]][idx[b]] = (Am[idx[a]][idx[b]] + wt * vec[a] * vec[b]) % p
        add_outer([0, 1, 2], [r[0], 0, r[2]], rng.randrange(2, p - 2))
        add_outer([1, 2, 3], [r[1] * s2 % p, r[2] * (s1 * c2 + c1 * s2) % p, r[3] * s1 % p], rng.randrange(2, p - 2))
        add_outer([2, 3, 4], [r[2] * s3 % p, r[3] * (s2 * c3 + c2 * s3) % p, r[4] * s2 % p], rng.randrange(2, p - 2))
        add_outer([3, 4, 5], [r[3], 0, r[5]], rng.randrange(2, p - 2))
        for j in range(4, n):
            idx = [j, (j + 1) % n, (j + 2) % n]
            if j == 4:
                vecs = [[0, r[idx[1]], r[idx[2]]], [r[idx[0]], 0, r[idx[2]]]]
            elif j == n - 1:
                vecs = [[r[idx[0]], r[idx[1]], 0], [r[idx[0]], 0, r[idx[2]]]]
            else:
                vecs = [[r[idx[0]], r[idx[1]], 0], [0, r[idx[1]], r[idx[2]]]]
            for _ in range(3):
                ca, cb = rng.randrange(2, p - 2), rng.randrange(2, p - 2)
                add_outer(idx, [(ca * vecs[0][k] + cb * vecs[1][k]) % p for k in range(3)], 1)
        return Am

    rng = Random(2026)
    results = {}
    for n in orders:
        eps = get_eps(n)
        def eval_clean_res(A_in, row_idx):
            b12, b13, b23, b24, b34 = A_in[1][2], A_in[1][3], A_in[2][3], A_in[2][4], A_in[3][4]
            pts_lam, v1_list, v2_list = [], [], []
            cand = 2
            while len(pts_lam) < 10:
                if cand != b23 % p:
                    pts_lam.append(cand)
                cand += 1
            for lv in pts_lam:
                L = [row[:] for row in A_in]
                for i, j in ((1,2),(2,1),(1,3),(3,1),(2,3),(3,2),(2,4),(4,2),(3,4),(4,3)):
                    L[i][j] = 0
                L[1][1] = (L[1][1] - b12 * b13 * inv(lv)) % p
                L[2][2] = (L[2][2] - b12 * lv * inv(b13) - b24 * (b23 - lv) * inv(b34)) % p
                L[3][3] = (L[3][3] - b13 * lv * inv(b12) - b34 * (b23 - lv) * inv(b24)) % p
                L[4][4] = (L[4][4] - b24 * b34 * inv(b23 - lv)) % p
                C = [[(eps[i] * eps[j] * L[i][j]) % p for j in range(n)] for i in range(n)]
                scale = lv * (b23 - lv) % p
                p1_v = scale * det_mod(C) % p
                cofs = [scale * (-1)**(row_idx + j) * det_mod([[C[r][c] for c in range(n) if c != j] for r in range(n) if r != row_idx]) % p for j in range(n)]
                lhs_c = pow(b23 - lv, 2, p) * (b12 * b13 * cofs[1]**2 - (b12 * inv(b13) * cofs[2]**2 + b13 * inv(b12) * cofs[3]**2) * lv**2) % p
                rhs_c = pow(lv, 2, p) * (b24 * b34 * cofs[4]**2 - (b24 * inv(b34) * cofs[2]**2 + b34 * inv(b24) * cofs[3]**2) * pow(b23 - lv, 2, p)) % p
                v1_list.append(p1_v)
                v2_list.append((lhs_c - rhs_c) * inv(pow(scale, 2, p)) % p)
            quartic = interp(pts_lam[:5], v1_list[:5])
            octic = interp(pts_lam[:9], v2_list[:9])
            for vals, poly, first in ((v1_list, quartic, 5), (v2_list, octic, 9)):
                for x, expected in zip(pts_lam[first:], vals[first:]):
                    assert sum(c * pow(x, k, p) for k, c in enumerate(poly)) % p == expected
            return poly_res_fixed(quartic, 4, octic, 8)

        for _ in range(3):
            pt_A = sample_meem_point(n, rng)
            assert eval_clean_res(pt_A, 0) == 0 and eval_clean_res(pt_A, 5) == 0

        if saved is None:
            pencils = ((i, random_band(n, 2, rng), random_band(n, 2, rng)) for i in range(1, 100))
        else:
            pencils = [(saved["attempt"], saved["A"], saved["B"])]
        for att, A0, B0 in pencils:
            N_pts = 32 * n
            xs, y0s, y5s, bps = [], [], [], []
            t_val = 0
            while len(xs) < N_pts + 1:
                At = [[(A0[i][j] + t_val * B0[i][j]) % p for j in range(n)] for i in range(n)]
                if all(At[i][j] != 0 for i, j in ((1,2),(1,3),(2,3),(2,4),(3,4))):
                    dc = pow(At[1][2] * At[1][3] * At[2][4] * At[3][4] % p, 12, p)
                    y0s.append(eval_clean_res(At, 0) * dc % p)
                    y5s.append(eval_clean_res(At, 5) * dc % p)
                    prod_all = 1
                    for i in range(n):
                        for d in (0, 1, 2):
                            prod_all = prod_all * At[i][(i + d) % n] % p
                    bps.append(prod_all)
                    xs.append(t_val)
                t_val += 1
            P0, P5 = interp(xs[:N_pts], y0s[:N_pts]), interp(xs[:N_pts], y5s[:N_pts])
            Pb = interp(xs[:3 * n + 1], bps[:3 * n + 1])
            G = p_gcd(P0, P5)
            while True:
                g2 = p_gcd(G, Pb)
                if len(g2) == 1:
                    break
                G, _ = p_divmod(G, g2)
            if saved is not None:
                assert xs == saved["interpolation_nodes"]
                for poly, value in zip((P0, P5), saved["holdout_values"]):
                    assert sum(c * pow(xs[-1], k, p) for k, c in enumerate(poly)) % p == value
                assert [y0s[-1], y5s[-1]] == saved["holdout_values"]
                assert G == saved["trimmed_gcd_coeffs_ascending"]
                rebuilt = [1]
                for factor in saved["factors"]:
                    coefficients = factor["coeffs_ascending"]
                    assert ddf_degrees(coefficients) == [(factor["d"], 1)]
                    for _ in range(factor["m"]):
                        rebuilt = p_mul(rebuilt, coefficients)
                assert rebuilt == G
            degs = ddf_degrees(G)
            mult1 = [d for d, m in degs if m == 1]
            if len(mult1) == 1:
                results[n] = (mult1[0], att)
                break
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="check the four stored pencils directly")
    if parser.parse_args().replay:
        path = Path(__file__).with_name("meem_pencils.json")
        for record in json.loads(path.read_text(encoding="utf-8")):
            p, n = record["prime"], record["order"]
            result = run_for_prime(p, (n,), saved=record)
            (degree,) = [factor["d"] for factor in record["factors"] if factor["m"] == 1]
            assert result[n] == (degree, record["attempt"])
            print(f"p={p}, n={n}: verified degree-{degree} factor")
    else:
        print("p=1009:", run_for_prime(1009, (6, 7, 8)))
        print("p=1013:", run_for_prime(1013, (6, 7)))
