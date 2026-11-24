"""Exact checkpoints for the revised manuscript. Python 3 + SymPy."""
import json
from pathlib import Path
from random import Random
from itertools import combinations

import sympy as sp
from base_checks import det_mod,random_band,interpolate_at_integers,winding_discriminant
from cp5 import membership
from types_and_counts import count_parts


def mask_equation(A,p):
    n=len(A);h=A[0][2]; jets=[]
    for t in (1,-1):
        M=[row.copy() for row in A]
        for i,j in ((0,n-2),(0,n-1),(1,n-1)):
            M[i][j]*=t;M[j][i]*=t
        def cof(i,j):
            return (-1)**(i+j)*det_mod([[M[r][c] for c in range(n) if c!=j]
                                         for r in range(n) if r!=i],p)
        jets.append((cof(0,0),2*h*cof(0,2)-det_mod(M,p),cof(2,2)))
    def bracket(i,j):
        return jets[0][i]*jets[1][j]-jets[1][i]*jets[0][j]
    return (h*h*bracket(0,2)**2-bracket(0,1)*bracket(1,2))%p


def degree_sample(n,mask,p=257):
    degree=4*n-2 if mask else n
    rng=Random(2461+n+int(mask));x=sp.Symbol("x")
    for attempt in range(1,650):
        A=random_band(n,2,rng);B=random_band(n,2,rng)
        if not mask:
            B=[[v if i==j else 0 for j,v in enumerate(row)] for i,row in enumerate(B)]
        def ev(t):
            M=[[a+t*b for a,b in zip(row,delta)] for row,delta in zip(A,B)]
            return mask_equation(M,p) if mask else winding_discriminant(M,2,p)
        f=sp.Poly.from_list(interpolate_at_integers([ev(t) for t in range(degree+1)],p)[::-1],
                            x,modulus=p)
        if f.degree()==degree and f.is_irreducible:
            assert all(int(f.eval(t))%p==ev(t) for t in (-1,degree+3))
            return dict(n=n,degree=degree,attempt=attempt,prime=p,A=A,B=B,
                        monic=[int(v)%p for v in f.monic().all_coeffs()])
    raise RuntimeError((n,mask))


def cyclotomic_check(n,k):
    x,y,t=sp.symbols('x y t');roots=list(range(3,3+k))
    T0,T=2,x
    for _ in range(1,n):
        T0,T=T,sp.expand(x*T-T0)
    f=sp.prod(x-r for r in roots)
    q=sp.prod(y-T.subs(x,r) for r in roots)
    out=sp.discriminant(f,x)
    for d in sp.divisors(n)[1:]:
        if d==2:
            R=sp.prod(a+b for a,b in combinations(roots,2))
        else:
            cycl=sp.cyclotomic_poly(d,t)
            # The resultant repeats each primitive inverse pair twice.
            twice=1
            for a,b in combinations(roots,2):
                quad=sp.expand((a*a+b*b-4)*t*t-a*b*(t**3+t)+(t*t+1)**2)
                twice*=sp.resultant(cycl,quad,t)
            R=sp.integer_nthroot(twice,2)[0]
            assert R*R==twice
        out*=R**2
    assert out==sp.discriminant(q,y)


def historical_checks():
    A=sp.Matrix(5,5,lambda i,j:[8,5,1,1,5][(j-i)%5])/8
    H=sp.Matrix(5,5,lambda i,j:-1 if (i-j)%5 in (1,4) else 1)
    out={}
    for name,M,is_cp in [('original',A,True)]:
        r=membership(M.tolist());assert r['cp']==is_cp;out[name]=r
    for j,ratio in [(2,sp.Rational(1,8)),(0,sp.Rational(1,5))]:
        S=sp.eye(5);S[4,j]=-ratio;M=S*A*S.T
        r=membership(M.tolist());assert not r['cp'] and (H*M).trace()<0
        out[f'delete_4_{j}']={'pairing':str((H*M).trace()),'decision':r}
    for name,M in [('negative_entry',sp.eye(5)),('indefinite',sp.eye(5))]:
        M[0,1]=M[1,0]=-1 if name=='negative_entry' else 2
        r=membership(M.tolist());assert not r['cp'];out[name]=r
    return out


if __name__=="__main__":
    for n,k in ((5,2),(6,2),(8,3),(9,4),(12,4)):
        cyclotomic_check(n,k)
    rng=Random(124)
    for n in range(5,11):
        B=sp.zeros(n)
        for j in range(n):
            for h in range(3):
                B[(j+h)%n,j]=rng.randrange(1,12)
        B[1,0]=0
        assert mask_equation([[int(v) for v in row] for row in (B*B.T).tolist()],257)==0
    report={'cyclotomic_orders':[5,6,8,9,12],'Dong_Anstreicher':historical_checks(),
            'masked_degree':[degree_sample(n,True) for n in (5,6,7,8)],
            'fixed_hopping_degree':[degree_sample(n,False) for n in (5,6,7,8,9)],
            'counts':{n:count_parts(n) for n in range(5,21)}}
    Path('revised_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print("Passed cyclotomic identities, published edge-deletion examples, mask and potential degrees, and labeled counts.")
