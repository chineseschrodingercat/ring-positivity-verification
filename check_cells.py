"""Exhaust all real angle/activity charts through a requested n, compare an independent
modular differential with the combinatorial rule for non-coordinate hypersurfaces."""
from itertools import product
from collections import Counter
from random import Random
import json
import sys
import numpy as np
from atlas import feasible,candidate_sums
from types_and_counts import mixed_data,count_parts

P=1009


def reduce_rank(a):
    a=np.array(a,dtype=np.int64)%P
    i=0
    for j in range(a.shape[1]):
        v=np.flatnonzero(a[i:,j])
        if not len(v): continue
        r=i+v[0]; a[[i,r]]=a[[r,i]]
        mult=a[i+1:,j]*pow(int(a[i,j]),-1,P)%P
        a[i+1:]=(a[i+1:]-mult[:,None]*a[i])%P
        i+=1
        if i==len(a): break
    return i


def check_one(a,s,active,seed):
    n=len(a); rng=Random(seed); pars={}; parity={}; deriv={}
    cos=[];sin=[];ds=[];dc=[]
    for i in range(n):
        if a[i]!='I' or i in parity: continue
        cc={i:1}; stack=[i]; forced=False
        while stack:
            v=stack.pop()
            for e,u in ((v,(v+1)%n),((v-1)%n,(v-1)%n)):
                if s[e]!='=' or a[u]!='I':continue
                if u in cc: forced |= cc[u]==cc[v]
                else: cc[u]=-cc[v];stack.append(u)
        h=1 if forced else rng.randrange(2,90)
        for j,sgn in cc.items():
            pars[j]=h if sgn==1 else pow(h,-1,P)
            parity[j]=sgn;deriv[j]=None if forced else i
    for i in range(n):
        if a[i]=='I':
            h=pars[i];inv=pow(1+h*h,-1,P)
            cos.append((1-h*h)*inv%P);sin.append(2*h*inv%P)
        else:cos.append(1 if a[i]=='0' else -1);sin.append(0)
    coords=[(i,(i+d)%n) for i in range(n) for d in range(3)]
    A=np.zeros((n,n),dtype=np.int64); params=[A.copy() for _ in range(2*n)]
    linear=[]; covered=set()
    for j in range(n):
        i=(j+1)%n; ids=[j,i,(j+2)%n];raw=[]; diffs={}
        if a[j]=='0':raw.append([1,1,0])
        if a[i]=='0':raw.append([0,1,1])
        if s[j] in '+=':raw.append([1,0,1])
        if a[j]!='0' and a[i]!='0' and s[j]=='-':
            cs=(cos[j]*cos[i]-sin[j]*sin[i])%P
            raw.append([sin[i],(sin[j]*cos[i]+cos[j]*sin[i])%P,sin[j]])
            diffs={j:[0,cs,cos[j]],i:[cos[i],cs,0]}
        raw=[v for v in raw if not any(x and not active[h] for x,h in zip(v,ids))]
        if not raw:diffs={}
        raw.extend([int(i0==v) for i0 in range(3)] for v,h in enumerate(ids) if not active[h])
        if not raw:continue
        G=np.array(raw,dtype=np.int64).T
        m=G.shape[1]; S=np.zeros((m,m),dtype=np.int64)
        for x in range(m):
            for y in range(x,m):
                S[x,y]=S[y,x]=rng.randrange(1,P)
                T=np.outer(G[:,x],G[:,y]);T=T if x==y else T+T.T
                emb=A.copy();emb[:]=0;emb[np.ix_(ids,ids)]=T
                linear.append([emb[u,v] for u,v in coords])
        V=G@S@G.T%P; A[np.ix_(ids,ids)]+=V
        for x in range(3):
            for y in range(3):
                if V[x,y]:covered.add(tuple(sorted((ids[x],ids[y]))))
        for h,dvec in diffs.items():
            if deriv[h] is not None:
                d=np.zeros_like(G);d[:,0]=np.array(dvec)*parity[h]
                mat=d@S@G.T;params[n+deriv[h]][np.ix_(ids,ids)]+=(mat+mat.T)%P
    if len(covered)<3*n:return -1
    for i in range(n):
        if active[i]:
            params[i][i,:]+=A[i,:];params[i][:,i]+=A[:,i]
    return reduce_rank(linear+[[m[u,v] for u,v in coords] for m in params])


def exhaustive(n):
    c=Counter(); mismatch=[]; components=set()
    for a in product('0IP',repeat=n):
        for s in product(*(candidate_sums(a[i],a[(i+1)%n]) for i in range(n))):
            if not feasible(a,s):continue
            optional=[i for i in range(n) if a[i-1]==a[i]=='P']
            for bits in product((0,1),repeat=len(optional)):
                active=[1]*n
                for i,v in zip(optional,bits):active[i]=v
                if not any(active):continue
                c['cells']+=1
                word=''.join('R' if a[j]!='I' and a[(j+1)%n]!='I' and s[j]!='+' else
                             'E' if a[j]==a[(j+1)%n]=='I' and s[j]=='-' else 'M' for j in range(n))
                valid=all(active) and all(not(a[j]=='I' or a[(j+1)%n]=='I') or
                           (s[j]!='=' and (s[j]=='+' or word[j]=='E')) for j in range(n))
                pure_e='R' not in word and word.count('M')<=1
                pure_r='I' not in a and 'MMM' not in (''.join(a)*2).replace('P','M')[:n+2]
                expected=bool(valid and (pure_e or pure_r or mixed_data(word)))
                rank=check_one(a,s,active,197)
                if expected and rank!=3*n-1:
                    rank=check_one(a,s,active,389)
                if (rank==3*n-1)!=expected:
                    mismatch.append((''.join(a),''.join(s),active,word,rank,expected))
                if rank==3*n-1:
                    c['hypersurface_charts']+=1
                    fixed=[]
                    for j in range(n):
                        if a[j]!='I':fixed.append((j,(j+1)%n,1 if a[j]=='P' else -1))
                        if word[j]!='E':fixed.append((j,(j+2)%n,1 if s[j]=='-' else -1))
                    label=min(tuple(g[i]*g[j]*val for i,j,val in fixed)
                              for tail in product((1,-1),repeat=n-1) for g in [(1,)+tail])
                    components.add((word,label))
    return dict(n=n,**c,components_including_coordinates=len(components)+2*n,
                predicted=sum(count_parts(n).values()),discrepancies=mismatch)


if __name__=='__main__':
    print(json.dumps([exhaustive(int(n)) for n in sys.argv[1:]],indent=2))
