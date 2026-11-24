"""Exact CP_5 membership. Usage: python cp5.py matrix.json (entries integers or rational strings).

The witness consists of a pencil, an irreducible root polynomial and its rational
isolating interval, a cycle, and an oriented normal. All decisions use Q arithmetic.
"""
import json
import sys
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, count, permutations, product
from math import factorial, lcm, prod
from random import Random

import sympy as sp

X=sp.Symbol("s")
P=lambda a:sp.Poly(a,X,domain=sp.QQ)
ZERO,ONE=P(0),P(1)
ENT=tuple(combinations_with_replacement(range(5),2))
CYCLES=[(0,)+p for p in permutations(range(1,5)) if p[0]<p[-1]]


def templates():
    result=[]
    for cycle in CYCLES:
        terms=[]
        for p in permutations(range(5)):
            sign=(-1)**sum(p[i]>p[j] for i,j in ENT)
            wind=sum((p[i]-i>=3)-(i-p[i]>=3) for i in range(5))
            edges=sum((i-p[i])%5 in (1,4) for i in range(5))
            indices=tuple(ENT.index(tuple(sorted((cycle[i],cycle[p[i]])))) for i in range(5))
            terms.append((indices,sign,wind,(-1)**edges))
        result.append(terms)
    return result


TERMS=templates()


def jets(values,terms,grad):
    entries=16 if grad else 1
    groups=[[0]*entries for _ in range(4)]  # winding 0,1,2, Horn
    for ids,sgn,w,h in terms:
        a=[values[i] for i in ids]
        val=prod(a)*sgn
        groups[3][0]+=h*val
        if w>=0:
            groups[w][0]+=val
        if grad:
            for j,i in enumerate(ids):
                dval=sgn*prod(a[:j])*prod(a[j+1:])
                groups[3][1+i]+=h*dval
                if w>=0:
                    groups[w][1+i]+=dval
    g,b,u,H=groups
    c=[v-2*r for v,r in zip(g,u)]
    D=[b[0]**2-4*u[0]*c[0]]
    D.extend(2*b[0]*b[i]-4*(u[i]*c[0]+u[0]*c[i]) for i in range(1,entries))
    return H,D


def interpolate(values):
    d=list(values); out=ZERO; term=ONE
    for i in range(len(values)):
        out+=term.mul_ground(sp.Rational(d[0],factorial(i)))
        term*=P(X-i)
        d=[b-a for a,b in zip(d,d[1:])]
    return out


def anchor_values(parameters):
    a=sp.zeros(5)
    for z,(i,j) in zip(parameters,ENT):
        a[i,j]=a[j,i]=z
    for i in range(5):
        a[i,i]+=sum(a[i,j] for j in range(5) if j!=i)
    return [int(a[i,j]) for i,j in ENT]


def anchors():
    rng=Random(7519)
    for _ in range(8):
        yield anchor_values([rng.randrange(1,18) for _ in ENT])
    for w in count(1):
        for v in product(range(1,w+1),repeat=15):
            if max(v)==w:
                yield anchor_values(v)


def pencil(A,C,idx,grad=False):
    results=[jets([(1-s)*c+s*a for a,c in zip(A,C)],TERMS[idx],grad) for s in range(11)]
    return [[interpolate([r[family][j] for r in results[:length]])
             for j in range(len(results[0][family]))]
            for family,length in ((0,6),(1,11))]


def events(A,C):
    out=[]; factors=set()
    for i in range(12):
        for family,polys in enumerate(pencil(A,C,i)):
            f=polys[0]
            if f.eval(0)==0:
                return None
            while f.eval(1)==0:
                f=f.exquo(P(X-1))
            for g,power in f.factor_list()[1]:
                g=g.monic()
                if power>1 or g in factors:
                    return None
                factors.add(g)
                for ab,_ in g.intervals(inf=0,sup=1):
                    out.append((i,family,g,ab))
    return out


class Root:
    def __init__(self,g,ab):
        self.g=g; self.a,self.b=ab

    def sign(self,f):
        f=f%self.g
        if f.is_zero:
            return 0
        coeff=f.all_coeffs()
        while self.a!=self.b:
            lo=hi=coeff[0]
            for c in coeff[1:]:
                v=(lo*self.a,lo*self.b,hi*self.a,hi*self.b)
                lo,hi=min(v)+c,max(v)+c
            if lo>0 or hi<0:
                return 1 if lo>0 else -1
            self.a,self.b=self.g.refine_root(self.a,self.b,steps=4)
        return int(sp.sign(f.eval(self.a)))


def is_copositive(M,root):
    g=root.g
    @lru_cache(maxsize=None)
    def minor(rows,cols):
        if not rows:
            return ONE
        return sum(((-1)**j*M[rows[0]][c]*minor(rows[1:],cols[:j]+cols[j+1:])%g
                    for j,c in enumerate(cols)),ZERO)%g
    for k in range(1,6):
        for S in combinations(range(5),k):
            ds=root.sign(minor(S,S))
            if not ds:
                continue
            for j in range(k):
                cj=S[:j]+S[j+1:]
                adj_sum=sum(((-1)**(i+j)*minor(S[:i]+S[i+1:],cj)
                             for i in range(k)),ZERO)
                if root.sign(adj_sum)*ds>=0:
                    break
            else:
                return False
    return True


def membership(matrix):
    M=sp.Matrix([[sp.Rational(v) for v in row] for row in matrix])
    if M.shape!=(5,5) or M!=M.T:
        raise ValueError("Expected a symmetric five-by-five matrix")
    if min(M)<0:
        return {"cp":False,"reason":"negative entry"}
    for k in range(1,6):
        for S in combinations(range(5),k):
            if M.extract(S,S).det()<0:
                return {"cp":False,"reason":"negative principal minor","indices":S}
    den=lcm(*(int(v.q) for v in M))
    A=[int(den*M[i,j]) for i,j in ENT]
    for idx,C in enumerate(anchors(),1):
        crossings=events(A,C)
        if crossings is not None:
            break
    cache={}
    for i,family,g,ab in crossings:
        root=Root(g,ab)
        if i not in cache:
            cache[i]=pencil(A,C,i,True)
        f,*gradient=cache[i][family]
        orientation=-root.sign(f.diff())
        assert orientation
        N=[[ZERO]*5 for _ in range(5)]
        for (a,b),value in zip(ENT,gradient):
            entry=(value%g).mul_ground(sp.Rational(orientation,1 if a==b else 2))
            N[a][b]=N[b][a]=entry
        if is_copositive(N,root):
            return {"cp":False,"reason":"copositive normal","scaled_input_denominator":den,
                    "anchor_upper":C,"cycle":CYCLES[i],"family":("Horn","Hildebrand")[family],
                    "root_polynomial_descending":[str(z) for z in g.all_coeffs()],
                    "interval":[str(root.a),str(root.b)],"normal_orientation":orientation}
    return {"cp":True,"reason":"no separating event","anchor_upper":C,"anchor_attempts":idx,
            "interior_real_events_tested":len(crossings)}


if __name__=="__main__":
    with open(sys.argv[1],encoding="utf-8-sig") as f:
        print(json.dumps(membership(json.load(f,parse_float=str)),indent=2))
