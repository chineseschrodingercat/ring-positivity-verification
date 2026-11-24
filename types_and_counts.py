"""Exact nonnegative cyclic-triple component counts. Types retain the vertex labels."""
from fractions import Fraction
from itertools import product
from math import comb
import sys


def binary_count(n):
    total=Fraction(2)
    for m in range(1,n//3+1):
        total+=Fraction(n,m)*(2*comb(n-2*m-1,m-1)-
                               (comb(m,n-3*m) if n-3*m<=m else 0))
    assert total.denominator==1
    return int(total)


def mixed_data(word):
    """Inputs are E (plane), R (rank one), M (clipped). Return the switching multiplicity."""
    n=len(word)
    if 'E' not in word or 'R' not in word or 'M' not in word:
        return 0
    start=word.index('M'); cut=word[start+1:]+word[:start]
    blocks=cut.split('M')
    if any(not b or len(set(b))!=1 or (b[0]=='R' and len(b)<2) for b in blocks):
        return 0
    joins=[i for i,b in enumerate(blocks) if b[0]==blocks[(i+1)%len(blocks)][0]=='E']
    if len(joins)>1:
        return 0
    if joins:
        i=joins[0]
        if len(blocks[i])==len(blocks[(i+1)%len(blocks)])==1:
            return 0
    return 1+int(all(len(b)==1 for b in blocks if b[0]=='E')
                 and any(len(b)>=4 for b in blocks if b[0]=='R'))


def count_parts(n):
    # Rational generating functions for labeled cyclic words. No consecutive E
    # blocks, or a single distinguished EE interface, with length-one pair removed.
    # [x^n](-n log(1-R(1+E))) includes the all-R words, which must be removed.
    import sympy as s
    x=s.Symbol("x"); E=x**2/(1-x); R=x**3/(1-x); E1=x**2
    # x*d/dx of a cyclic logarithm counts labels. For one marked interface,
    # distinguish its initial vertex and multiply by n.
    def mixed_cycle(e,r):
        f=1-r*(1+e)
        return x*s.diff(1-r,x)/(1-r)-x*s.diff(f,x)/f
    base=mixed_cycle(E,R)
    extra=mixed_cycle(E1,R)-mixed_cycle(E1,x**3+x**4)
    marked=(E**2-E1**2)*R/(1-R*(1+E))
    def coefficient(f):
        num,den=s.fraction(s.cancel(f))
        a=s.Poly(num,x); b=s.Poly(den,x); cs=[]
        for i in range(n+1):
            cs.append((a.nth(i)-sum(b.nth(j)*cs[i-j]
                                    for j in range(1,min(i,b.degree())+1)))/b.nth(0))
        return int(cs[-1])
    m0=coefficient(base); m1=n*coefficient(marked); labels=coefficient(extra)
    # The all-E case with one clipped window contributes n further labels.
    return dict(coordinate=2*n,binary=binary_count(n),balanced=1,
                single_mask=n,mixed_no_EE=m0,mixed_one_EE=m1,extra_switching=labels)


if __name__=="__main__":
    import json
    for n in map(int,sys.argv[1:] or range(5,15)):
        c=count_parts(n)
        if n<=10:
            direct=sum(mixed_data(''.join(w)) for w in product('ERM',repeat=n))
            assert direct==sum(c[k] for k in ('mixed_no_EE','mixed_one_EE','extra_switching'))
        print(json.dumps({"n":n,"N":sum(c.values()),**c}))
