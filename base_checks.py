"""Exact arithmetic checkpoints for Cyclic_Gram_boundaries.tex (Python + SymPy)."""
import json
import math
from functools import lru_cache
from itertools import permutations, product
from random import Random

import sympy as sp


def det_mod(matrix, p):
    a = [[entry % p for entry in row] for row in matrix]
    n = len(a)
    det = 1
    for i in range(n):
        pivot = next((j for j in range(i, n) if a[j][i]), None)
        if pivot is None:
            return 0
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            det = -det
        det = det * a[i][i] % p
        inv = pow(a[i][i], -1, p)
        for j in range(i + 1, n):
            mul = a[j][i] * inv % p
            if mul:
                for h in range(i + 1, n):
                    a[j][h] = (a[j][h] - mul*a[i][h]) % p
    return det


def random_band(n, k, rng):
    a = [[0]*n for _ in range(n)]
    for i in range(n):
        for h in range(k+1):
            j = (i+h) % n
            a[i][j] = a[j][i] = rng.randrange(-6, 7)
    return a


def twisted(matrix, k, t):
    n = matrix.rows
    out = matrix.copy()
    for i in range(k):
        for j in range(n-k+i, n):
            out[i,j] *= t
            out[j,i] /= t
    return out


@lru_cache(maxsize=None)
def disc_plan(k, p):
    ts = list(range(1,k+2))
    vs = [(t+pow(t,-1,p)) % p for t in ts]
    assert len(set(vs)) == k+1
    inverse = sp.Matrix([[pow(v,j,p) for j in range(k+1)] for v in vs]).inv_mod(p)
    x = sp.Symbol("x")
    coeff = sp.symbols(f"c0:{k+1}")
    disc = sp.discriminant(sum(c*x**j for j,c in enumerate(coeff)),x)
    return ts, [[int(v) for v in row] for row in inverse.tolist()], sp.lambdify(coeff,disc)


def winding_discriminant(matrix, k, p):
    ts, inverse, disc = disc_plan(k, p)
    n = len(matrix)
    dets = []
    for t in ts:
        a = [row.copy() for row in matrix]
        for i in range(k):
            for j in range(n-k+i,n):
                a[i][j] *= t
                a[j][i] *= pow(t,-1,p)
        dets.append(det_mod(a,p))
    coeff = [sum(u*v for u,v in zip(row,dets))%p for row in inverse]
    return int(disc(*coeff)) % p


def interpolate_at_integers(values, p):
    # Newton forward differences give coefficients in the ordinary power basis.
    differences = values.copy()
    basis = [1]
    result = [0]*len(values)
    for j in range(len(values)):
        scale = differences[0]*pow(math.factorial(j),-1,p)%p
        for i,c in enumerate(basis):
            result[i] = (result[i]+scale*c)%p
        differences = [(b-a)%p for a,b in zip(differences,differences[1:])]
        basis = [((basis[i-1] if i else 0)-(j*basis[i] if i<len(basis) else 0))%p
                 for i in range(len(basis)+1)]
    return result


def line_certificate(n, k, p=257):
    degree = 2*n*(k-1)
    rng = Random(70900+100*n+k)
    x = sp.Symbol("x")
    for attempt in range(1, 600):
        A, B = random_band(n,k,rng), random_band(n,k,rng)
        def value(t):
            m = [[a+t*b for a,b in zip(row,direction)] for row,direction in zip(A,B)]
            return winding_discriminant(m,k,p)
        coeff = interpolate_at_integers([value(t) for t in range(degree+1)],p)
        f = sp.Poly.from_list(coeff[::-1],x,modulus=p)
        if f.degree() != degree or not f.is_irreducible:
            continue
        assert all(int(f.eval(t))%p == value(t) for t in (-1,degree+2))
        return {"n":n,"k":k,"degree":degree,"prime":p,"line":attempt,
                "A":A,"B":B,"monic_descending":[int(c)%p for c in f.monic().all_coeffs()]}
    raise ArithmeticError(f"No certificate located for {(n,k)}")


def clipped_example(n):
    # Half-angle tangents 2, 3, 1/4, ... . Only the first sum exceeds pi.
    Q=sp.Rational
    ts = [Q(2),Q(3)]+[Q(1,4)]*(n-2)
    co = [(1-t*t)/(1+t*t) for t in ts]
    si = [2*t/(1+t*t) for t in ts]
    W=sp.zeros(n)
    B=sp.zeros(n)
    for j in range(n):
        i=(j+1)%n
        l=(j+2)%n
        W[j,j]=1
        W[i,j]=W[j,i]=-co[j]
        W[l,j]=W[j,l]=-1 if ts[j]*ts[i] >= 1 else co[j]*co[i]-si[j]*si[i]
        entries = ((1,0,1) if j==0 else (40,13,51) if j==1 else
                   (17,9,10) if j==n-1 else (17,30,17))
        B[j,j],B[i,j],B[l,j] = entries
    A=B*B.T
    assert (A*W).trace()==0
    alpha=sp.prod(B[j,j] for j in range(n))
    gamma=sp.prod(B[(j+2)%n,j] for j in range(n))
    assert alpha/gamma==Q(4,3)
    beta=B.det()-alpha-gamma
    delta=(alpha-gamma)**2*(beta*beta-4*alpha*gamma)
    assert delta and B.det() and alpha-beta+gamma
    coords=[(i,(i+d)%n) for i in range(n) for d in range(3)]
    jac=[]
    for j in range(n):
        for h in range(3):
            if j==0 and h==1:
                continue
            U=sp.zeros(n); U[(j+h)%n,j]=1
            deriv=U*B.T+B*U.T
            jac.append([deriv[a,b] for a,b in coords])
    p=1009
    minors=[]
    for drop in range(3*n):
        sub=[[int(v) for i,v in enumerate(row) if i!=drop] for row in jac]
        val=det_mod(sub,p)
        if val:
            minors=[drop,val]
            break
    assert minors
    return {"order":n,"supporting_pairing":0,"endpoint_ratio":"4/3",
            "Hplus":int(A.det()),"Hminus":int(twisted(A,2,sp.S.NegativeOne).det()),
            "discriminant":int(delta), "Jacobian_minor_mod_1009":minors}


def five_check():
    n=5
    rng=Random(19)
    A=sp.Matrix(random_band(5,2,rng))
    H=sp.Matrix(5,5,lambda i,j:-1 if (i-j)%5 in (1,4) else 1)
    S=sp.diag(1,-1,1,-1,1)
    assert H.multiply_elementwise(A)==S*twisted(A,2,sp.S.NegativeOne)*S
    x=sp.symbols("x:15")
    upper=[(i,j) for i in range(n) for j in range(i,n)]
    coords=dict(zip(upper,x)); groups=[0]*5
    for perm in permutations(range(n)):
        wind=sum((j-i>=3)-(i-j>=3) for i,j in enumerate(perm))
        parity=sum(perm[i]>perm[j] for i,j in upper)
        groups[2+wind]+=(-1)**parity*sp.prod(coords[min(i,j),max(i,j)] for i,j in enumerate(perm))
    u,b,c=groups[4],groups[3],groups[2]-2*groups[4]
    decic=sp.Poly(b*b-4*u*c,*x)
    assert decic.total_degree()==10 and len(decic.terms())==313
    return {"degree":10,"monomials":313,"horn_gauge":True}


def binary_components(n):
    classes=set()
    for angles in product((0,1),repeat=n):  # 1 represents pi
        if any(all(angles[(j+i)%n] for i in range(3)) for j in range(n)):
            continue
        regular=[not(angles[j] and angles[(j+1)%n]) for j in range(n)]
        triples=tuple(j for j,x in enumerate(regular) if not x)
        eta=(-1)**(n-sum(angles))
        assert all(regular[j] or regular[j-1] for j in range(n))
        classes.add((triples,eta))
    count=2
    for m in range(1,n//3+1):
        count+=2*n*math.comb(n-2*m-1,m-1)//m
        if n-3*m<=m:
            count-=n*math.comb(m,n-3*m)//m
    assert len(classes)==count
    return count


def fourier_collision(n,k,p):
    # Two distinct spatial modes coalesce at one phase. Check the double
    # multiplicity along a circulant pencil and the vanishing conormal exactly.
    generator=int(sp.primitive_root(p))
    xi=pow(generator,(p-1)//n,p)
    assert pow(xi,n,p)==1 and xi!=1
    r1=generator
    r2=r1*xi%p
    z=pow(r1,n,p)
    assert z not in (1,p-1)
    y1=(r1+pow(r1,-1,p))%p
    y2=(r2+pow(r2,-1,p))%p
    assert y1!=y2
    assert sum(pow(r1*pow(r2,-1,p)%p,j,p) for j in range(n))%p==0
    # Map folded scalar eigenvalue via Chebyshev's recurrence.
    def fold(y):
        a,b=2,y
        for _ in range(1,n):
            a,b=b,(y*b-a)%p
        return b
    excluded={fold(y1),2,p-2}
    more=[]
    for h in range(3,p):
        if fold(h) not in excluded:
            more.append(h); excluded.add(fold(h))
            if len(more)==k-2:
                break
    degree=2*n*(k-1)
    values=[]
    for t in range(degree+1):
        roots=[fold((y1+t)%p),fold(y2)]+[fold(h) for h in more]
        values.append(math.prod((u-v)**2 for i,u in enumerate(roots) for v in roots[i+1:])%p)
    cs=interpolate_at_integers(values,p)
    assert cs[0]==cs[1]==0 and cs[2]!=0
    return {"n":n,"k":k,"p":p,"r1":r1,"r2":r2,"coefficient_of_t_squared":cs[2]}


if __name__=="__main__":
    report={"SymPy":sp.__version__, "five":five_check(),
            "binary_determinant_counts":{n:binary_components(n) for n in range(5,15)},
            "folded_collisions":[fourier_collision(7,3,337),fourier_collision(9,4,109)],
            "clipped_positive_points":[clipped_example(n) for n in (5,6,7,8)],
            "irreducible_lines":[line_certificate(n,k) for n,k in ((7,3),(8,3),(9,4))]}
    print(json.dumps(report,indent=2))
