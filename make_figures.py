"""Regenerate the figures of 'Positivity and complete positivity on a ring'.

Usage:  python make_figures.py
Writes fig_ring.pdf, fig_word.pdf, fig_cp5_slice.pdf and fig_growth.pdf in the
current folder.  The growth figure uses the generating function of
Proposition (Count and growth); SymPy is needed for that figure only.

Figure 3 is exact. It shows the plane of 5x5 circulant matrices with unit
diagonal, ordinary-edge entry a and chord entry b.  The doubly nonnegative
region is a >= 0, b >= 0 and 1 + 2a cos(2 pi k/5) + 2b cos(4 pi k/5) >= 0 for
k = 1, 2.  For the identity cyclic order the Horn quintic contains the line
1 - 2a + 2b = 0 (trivial-character factor of det(H o A)), and on this plane the
Hildebrand decic equals disc(f) R_5(f)^2 with f(y) = b y^2 + a y + 1 - 2b, where
R_5 = 0 is the pair of conics a^2 - (2+c) b(1-2b) + (c^2-4) b^2 = 0 with
c = 2cos(2pi/5), 2cos(4pi/5).  The pentagram order (0,2,4,1,3) exchanges a and b.
The circulant Gram points of the equal-angle Hildebrand faces form the arc
b = 1/(2 + 4cos^2 t), a = 4b cos t, 0 <= t <= pi/5, which lies on disc(f) = 0.
The shaded region is the convex hull of I, J, these arcs and the two Horn
segments, all completely positive; every other doubly nonnegative point fails
a Horn test 1 - 2a + 2b >= 0 or an equal-angle test
1 - 2a cos t + 2b cos 2t >= 0 of one of the two orders.
"""
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, Polygon
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 10, "axes.linewidth": 0.8})
BLUE, RED, GREEN, ORANGE, PURPLE = "#2b6cb0", "#c53030", "#2f855a", "#dd6b20", "#6b46c1"


# ---------------------------------------------------------------- Figure 1
def fig_ring(path="fig_ring.pdf", n=10):
    fig, ax = plt.subplots(figsize=(4.0, 4.2))
    ang = [math.pi / 2 - 2 * math.pi * i / n for i in range(n)]
    P = np.array([[math.cos(a), math.sin(a)] for a in ang])
    for i in range(n):
        ax.plot(*zip(P[i], P[(i + 2) % n]), color="0.82", lw=0.9, zorder=1)
    for i in range(n):
        ax.plot(*zip(P[i], P[(i + 1) % n]), color="0.35", lw=1.3, zorder=2)
    for i, j in ((0, n - 2), (0, n - 1), (1, n - 1)):          # entries crossing the cut
        ax.plot(*zip(P[i], P[j]), color=ORANGE, lw=2.2, zorder=3)
    ax.scatter(P[:, 0], P[:, 1], s=34, color="black", zorder=4)
    for i in range(n):
        ax.text(1.13 * P[i, 0], 1.13 * P[i, 1], str(i), ha="center", va="center", fontsize=9)
    for j, col, r in ((0, BLUE, 1.30), (n - 2, RED, 1.44), (n - 1, GREEN, 1.58)):
        th_end = math.degrees(ang[j])
        th_start = math.degrees(ang[(j + 2) % n])
        while th_end < th_start:
            th_end += 360
        ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=th_start - 5, theta2=th_end + 5, color=col, lw=2.0))
        am = math.radians(th_start - 9) if j != n - 2 else math.radians(th_end + 9)
        ax.text((r + 0.02) * math.cos(am), (r + 0.02) * math.sin(am), rf"$T_{{{j}}}$",
                color=col, ha="center", va="center", fontsize=10)
    cut = (ang[0] + ang[n - 1] + 2 * math.pi) / 2
    ax.plot([0.12 * math.cos(cut), 1.78 * math.cos(cut)], [0.12 * math.sin(cut), 1.78 * math.sin(cut)],
            ls=(0, (4, 3)), color="black", lw=1.0, zorder=5)
    ax.text(1.86 * math.cos(cut), 1.86 * math.sin(cut), r"cut: $\times\,t^{\pm1}$",
            ha="center", va="bottom", fontsize=9)
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.35, 2.05); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig_word(path="fig_word.pdf"):
    word = "RRRMEEMEMRRM"
    fill = {"R": BLUE, "E": ORANGE, "M": "#cbd5e0"}
    fig, ax = plt.subplots(figsize=(6.6, 1.65))
    for i, ch in enumerate(word):
        ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=fill[ch], edgecolor="white", lw=2))
        ax.text(i + 0.5, 0.5, ch, ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if ch != "M" else "black")
        ax.text(i + 0.5, -0.27, rf"$T_{{{i}}}$", ha="center", va="center", fontsize=8)

    def span(x0, x1, label):
        ax.annotate("", xy=(x0 + 0.06, 1.18), xytext=(x1 - 0.06, 1.18),
                    arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.3"))
        ax.text((x0 + x1) / 2, 1.34, label, ha="center", va="bottom", fontsize=8)

    span(0, 3, r"$R$ block, length 3")
    span(4, 6, r"$E$ block, length 2")
    span(7, 8, r"$E$, length 1")
    span(9, 11, r"$R$ block, length 2")
    ax.annotate("plane–plane join", xy=(6.5, -0.48), xytext=(6.5, -0.95), ha="center", va="center",
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax.set_xlim(-0.2, 12.2); ax.set_ylim(-1.15, 1.7); ax.axis("off")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Figure 3
C1, C2 = 2 * math.cos(2 * math.pi / 5), 2 * math.cos(4 * math.pi / 5)


def lam(a, b, k):
    return 1 + 2 * a * math.cos(2 * math.pi * k / 5) + 2 * b * math.cos(4 * math.pi * k / 5)


def disc_f(a, b):
    return a * a - 4 * b * (1 - 2 * b)


def r5_factor(a, b, c):
    return a * a - (2 + c) * b * (1 - 2 * b) + (c * c - 4) * b * b


def hildebrand_arc(m=200):
    t = np.linspace(0, math.pi / 5, m)
    b = 1 / (2 + 4 * np.cos(t) ** 2)
    return 4 * b * np.cos(t), b


def cp_region_boundary():
    """Boundary of the CP region in the circulant plane, traversed counterclockwise.

    The identity order cuts the sliver at (1/g, 0) and the pentagram order, which
    exchanges a and b, cuts the mirror sliver at (0, 1/g)."""
    a_arc, b_arc = hildebrand_arc()
    seg = [(0.5 + s * (a_arc[0] - 0.5), s * b_arc[0]) for s in np.linspace(0, 1, 30)]
    pts = [(0.0, 0.0)] + seg + list(zip(a_arc[1:], b_arc[1:])) + [(1.0, 1.0)]
    pts += list(zip(b_arc[::-1], a_arc[::-1])) + [(q, p) for p, q in seg[::-1]] + [(0.0, 0.0)]
    return np.array(pts)


def dnn_polygon():
    g = (1 + 5 ** 0.5) / 2
    return np.array([(0, 0), (1 / g, 0), (1, 1), (0, 1 / g)])


def fig_slice(path="fig_cp5_slice.pdf"):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), gridspec_kw={"width_ratios": [1, 1.05]})
    A, B = np.meshgrid(np.linspace(-0.05, 1.1, 700), np.linspace(-0.05, 1.1, 700))
    inside_dnn = (A >= 0) & (B >= 0) & (lam(A, B, 1) >= 0) & (lam(A, B, 2) >= 0)
    a_arc, b_arc = hildebrand_arc()
    for panel, (ax, (xl, yl)) in enumerate(zip(axes, (((-0.04, 1.06), (-0.04, 1.06)), ((0.46, 0.76), (-0.015, 0.285))))):
        orders = ((A, B), (B, A)) if panel == 0 else ((A, B),)
        ax.add_patch(Polygon(dnn_polygon(), closed=True, facecolor="#fde8e8", edgecolor="none", zorder=0))
        ax.add_patch(Polygon(cp_region_boundary(), closed=True, facecolor="#e2e8f0", edgecolor="none", zorder=1))
        ax.add_patch(Polygon(dnn_polygon(), closed=True, facecolor="none", edgecolor="0.2", lw=1.0,
                             ls=(0, (4, 3)), zorder=2))
        for c in (C1, C2):
            for P_, Q_ in orders:
                ax.contour(A, B, np.where(inside_dnn, r5_factor(P_, Q_, c), np.nan), levels=[0], colors=[PURPLE],
                           linewidths=0.8, linestyles="dotted", zorder=3)
        for P_, Q_ in orders:
            ax.contour(A, B, np.where(inside_dnn, disc_f(P_, Q_), np.nan), levels=[0], colors=[RED],
                       linewidths=0.7, alpha=0.5, zorder=3)
        hx = np.linspace(0.5, 1.06, 50)
        ax.plot(hx, hx - 0.5, color=BLUE, lw=0.7, alpha=0.5, zorder=3)
        ax.plot(hx - 0.5, hx, color=BLUE, lw=0.7, alpha=0.5, zorder=3)
        ax.plot([0.5, a_arc[0]], [0.0, b_arc[0]], color=BLUE, lw=2.2, zorder=4, solid_capstyle="round")
        ax.plot([0.0, b_arc[0]], [0.5, a_arc[0]], color=BLUE, lw=2.2, zorder=4, solid_capstyle="round")
        ax.plot(a_arc, b_arc, color=RED, lw=2.2, zorder=4, solid_capstyle="round")
        ax.plot(b_arc, a_arc, color=RED, lw=2.2, zorder=4, solid_capstyle="round")
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
        ax.set_xlabel(r"edge entry $a$"); ax.tick_params(labelsize=8)
    axes[0].set_ylabel(r"chord entry $b$")
    axes[0].add_patch(Rectangle((0.46, -0.015), 0.30, 0.30, fill=False, lw=0.7, ec="0.4", zorder=5))
    axes[0].set_title(r"circulants with unit diagonal", fontsize=9)
    axes[1].set_title(r"enlarged corner", fontsize=9)
    axes[1].annotate(r"$\theta=\pi/5$", xy=(a_arc[-1], b_arc[-1]), xytext=(0.715, 0.255), fontsize=8,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
    axes[1].annotate(r"$\theta=0$", xy=(a_arc[0], b_arc[0]), xytext=(0.6, 0.2), fontsize=8,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
    handles = [Line2D([], [], color=BLUE, lw=2.2, label=r"Horn quintics"),
               Line2D([], [], color=RED, lw=2.2, label=r"Hildebrand decics"),
               Line2D([], [], color=PURPLE, lw=0.9, ls="dotted", label=r"doubled factors $R_5=0$"),
               Line2D([], [], color="0.2", lw=1.0, ls=(0, (4, 3)), label=r"$\partial\,\mathcal{DNN}_5$"),
               plt.Rectangle((0, 0), 1, 1, fc="#e2e8f0", label=r"$\mathcal{CP}_5$"),
               plt.Rectangle((0, 0), 1, 1, fc="#fde8e8", label=r"$\mathcal{DNN}_5\setminus\mathcal{CP}_5$")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def component_counts(nmax=24):
    """N(n) for 5 <= n <= nmax from the generating function (eq:count), with B_n."""
    import sympy as sp
    from math import comb
    x = sp.symbols("x")
    E = x**2/(1-x); E1 = x**2; R = x**3/(1-x); R0 = x**3 + x**4
    L = lambda e, r: x*sp.diff(sp.log((1-r)/(1-r*(1+e))), x)
    s1 = sp.series(L(E, R) + L(E1, R) - L(E1, R0), x, 0, nmax+1).removeO()
    s2 = sp.series((E**2 - E1**2)*R/(1 - R*(1+E)), x, 0, nmax+1).removeO()
    def B(n):
        a = sum(sp.Rational(n, m)*comb(n-2*m-1, m-1) for m in range(1, n//3+1) if n-2*m-1 >= 0)
        b = sum(sp.Rational(n, m)*(comb(m, n-3*m) if 0 <= n-3*m <= m else 0) for m in range(1, n//3+1))
        return 2*(1+a) - b
    N = {n: int(2*n + B(n) + 1 + n + s1.coeff(x, n) + n*s2.coeff(x, n)) for n in range(5, nmax+1)}
    tau = min(r for r in sp.Poly(x**5 - x**4 + x**3 - x**2 + 2*x - 1, x).nroots() if abs(sp.im(r)) < 1e-12 and 0 < sp.re(r) < 1)
    tau = float(sp.re(tau))
    F = R*(1+E)
    cstar = float(((E**2 - E1**2)*R/(x*sp.diff(F, x))).subs(x, tau))
    return N, 1/tau, cstar


def fig_growth(path="fig_growth.pdf", nmax=24):
    N, rho, cstar = component_counts(nmax)
    ns = sorted(N)
    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    face = [n for n in ns if n >= 7]
    ax.semilogy(face, [N[n] for n in face], "o", color=BLUE, ms=4.5,
                label=r"$K^+_{n,2}$: components $N(n)$")
    ax.semilogy([5, 6], [N[5], N[6]], "o", mfc="white", mec=BLUE, ms=4.5)
    grid = np.linspace(5, nmax, 200)
    ax.semilogy(grid, (cstar*grid + 1)*rho**grid, color=BLUE, lw=0.9, ls=(0, (4, 3)),
                label=rf"$(c_*n+1)\,\rho^n$, $\rho={rho:.4f}$")
    ax.semilogy(ns, [3]*len(ns), "s", color=RED, ms=4,
                label=r"$K_{n,2}$: three components")
    ax.set_xlabel(r"order $n$"); ax.set_ylabel("irreducible components")
    ax.set_xticks(range(5, nmax+1, 2))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return N, rho, cstar


if __name__ == "__main__":
    fig_ring()
    fig_word()
    fig_slice()
    N, rho, cstar = fig_growth()
    print("wrote fig_ring.pdf, fig_word.pdf, fig_cp5_slice.pdf, fig_growth.pdf")
    print("N(5..24) =", [N[n] for n in sorted(N)])
    print("rho =", rho, " c_* =", cstar, " N(24)/rho^24 - 24 c_* =", N[24]/rho**24 - 24*cstar)
