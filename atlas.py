"""Rational maps for the finite nonnegative triple boundary atlas in the note.

python atlas.py 5
python atlas.py 5 --angles IIIII --sums +---- --active 11111
python atlas.py 5 --angles IIIII --sums +---- --active 11111 --rank

The first command counts feasible parameter cells (not distinct components).
The second prints the map generators for one cell. 0,I,P denote 0, interior, pi.
The sums compare consecutive angles with pi. The entries in --active indicate
positive dual diagonals. An omitted inactive-edge angle can always be fixed to pi.
"""
import argparse
import json
from itertools import product

import sympy as sp


def candidate_sums(a,b):
    if a == b == "I":
        return "-=+"
    if {a,b} == {"0","P"}:
        return "="
    return "-" if "0" in (a,b) else "+"


def feasible(angles,sums):
    """Strict signed comparisons of the centered angles form a partial order."""
    n=len(angles)
    # Labels +/-i encode the two signs of a centered angle; +/-(n+1) are +/-1.
    top=n+1
    nodes=tuple(range(1,top+1))+tuple(range(-top,0))
    parent={v:v for v in nodes}
    def root(v):
        while parent[v]!=v:
            v=parent[v]
        return v
    def merge(a,b):
        parent[root(a)] = root(b)
        parent[root(-a)] = root(-b)
    edges=[(-top,top)]
    for i,(angle,sign) in enumerate(zip(angles,sums),1):
        j=i%n+1
        if angle in ("0","P"):
            merge(i,top if angle=="P" else -top)
        else:
            for v in (i,-i):
                edges.extend([(-top,v),(v,top)])
        if sign=="=":
            merge(i,-j)
        elif sign=="+":
            edges.extend([(-j,i),(-i,j)])
        else:
            edges.extend([(i,-j),(j,-i)])
    graph={root(v):set() for v in nodes}
    indegree={v:0 for v in graph}
    for a,b in edges:
        a,b=root(a),root(b)
        if a==b:
            return False
        if b not in graph[a]:
            graph[a].add(b)
            indegree[b]+=1
    stack=[v for v in graph if indegree[v]==0]
    processed=0
    while stack:
        a=stack.pop()
        processed+=1
        for b in graph[a]:
            indegree[b]-=1
            if indegree[b]==0:
                stack.append(b)
    # A topological height can be antisymmetrized by h(v)-h(-v), then rescaled.
    return processed==len(graph)


def angle_charts(angles,sums):
    n=len(angles)
    half={}
    params=[]
    for i in range(n):
        if angles[i]!="I" or i in half:
            continue
        parity={i:0}
        stack=[i]
        odd=False
        while stack:
            v=stack.pop()
            for edge,u in ((v,(v+1)%n),((v-1)%n,(v-1)%n)):
                if sums[edge]!="=" or angles[u]!="I":
                    continue
                if u in parity:
                    odd |= parity[u]==parity[v]
                else:
                    parity[u]=1-parity[v]
                    stack.append(u)
        t=sp.S.One if odd else sp.Symbol(f"u{i}")
        if not odd:
            params.append(t)
        for v,inv in parity.items():
            half[v]=1/t if inv else t
    cs=[]
    for i,a in enumerate(angles):
        if a=="I":
            t=half[i]
            cs.append(((1-t*t)/(1+t*t),2*t/(1+t*t)))
        else:
            cs.append((1 if a=="0" else -1,0))
    return params,cs


def make_map(angles,sums,active):
    """Return rational packed-window generators; A is the sum of G S G^T."""
    n=len(angles)
    assert len(active)==len(sums)==n and "1" in active
    if not feasible(angles,sums):
        raise ValueError("Empty angle cell")
    params,cs=angle_charts(angles,sums)
    scales={i:sp.Symbol(f"r{i}") for i in range(n) if active[i]=="1"}
    windows=[]
    for j in range(n):
        i=(j+1)%n
        indices=[j,i,(j+2)%n]
        raw=[]
        if angles[j]=="0":
            raw.append((1,1,0))
        if angles[i]=="0":
            raw.append((0,1,1))
        if sums[j] in "+=":
            raw.append((1,0,1))
        if angles[j]!="0" and angles[i]!="0" and sums[j]=="-":
            cj,sj=cs[j]; ci,si=cs[i]
            raw.append((si,sj*ci+cj*si,sj))
        local=[]
        for v in raw:
            if all(not x or h in scales for x,h in zip(v,indices)):
                local.append(sp.Matrix([scales.get(h,0)*x for h,x in zip(indices,v)]))
        local.extend(sp.eye(3)[:,h] for h,i0 in enumerate(indices) if i0 not in scales)
        windows.append(sp.Matrix.hstack(*local) if local else sp.zeros(3,0))
    return list(scales.values())+params,windows


def count_cells(n):
    angle_cells=0
    active_cells=0
    for a in product("0IP",repeat=n):
        options=[candidate_sums(a[j],a[(j+1)%n]) for j in range(n)]
        for signs in product(*options):
            if not feasible(a,signs):
                continue
            angle_cells+=1
            # The dominated normal permits both edges incident to an inactive
            # vertex to use pi. Including all other charts is redundant.
            optional=sum(a[(i-1)%n]==a[i]=="P" for i in range(n))
            active_cells+=(1<<optional)-int(optional==n)
    return {"n":n,"angle_cells":angle_cells,"canonical_angle_activity_cells":active_cells,
            "coordinate_facets":2*n}


def chart_rank(n,params,windows):
    """Use a nonsingular exact test minor if conclusive, else symbolic generic rank."""
    coords=[(j,(j+d)%n) for j in range(n) for d in range(3)]
    A=sp.zeros(n)
    coeffs=[]
    for j,G in enumerate(windows):
        K=sp.zeros(G.cols)
        for a in range(G.cols):
            for b in range(a,G.cols):
                s=sp.Symbol(f"s{j}_{a}_{b}")
                K[a,b]=K[b,a]=s
                coeffs.append(s)
        local=G*K*G.T
        for a in range(3):
            for b in range(3):
                A[(j+a)%n,(j+b)%n]+=local[a,b]
    variables=params+coeffs
    jac=sp.Matrix([A[a,b] for a,b in coords]).jacobian(variables)
    values={v:i+2 for i,v in enumerate(variables)}
    rank=int(jac.xreplace(values).to_DM().rank())
    if rank==3*n-1:
        return {"rank":rank,"certificate":"exact rational evaluation and boundary dimension"}
    return {"rank":int(jac.to_DM().rank()),"certificate":"rank over the rational function field"}


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n",type=int)
    ap.add_argument("--angles"); ap.add_argument("--sums"); ap.add_argument("--active")
    ap.add_argument("--rank",action="store_true")
    args=ap.parse_args()
    if args.n<5:
        ap.error("n must be at least five")
    if args.angles is None:
        if args.sums or args.active:
            ap.error("specify all three cell arguments")
        print(json.dumps(count_cells(args.n),indent=2))
    else:
        if any(not s or len(s)!=args.n or not set(s)<=set(alphabet)
               for s,alphabet in ((args.angles,"0IP"),(args.sums,"-=+"),(args.active,"01"))):
            ap.error("invalid cell arguments")
        params,windows=make_map(args.angles,args.sums,args.active)
        if args.rank:
            print(json.dumps(chart_rank(args.n,params,windows),indent=2))
        else:
            print("parameters:",params)
            for j,G in enumerate(windows):
                print(f"T{j}:",G.tolist())
