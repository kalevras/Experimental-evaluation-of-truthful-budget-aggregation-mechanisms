"""Pilot hill-climbing with aggregated voter types, following the supplied 8-page protocol.
No worst-case upper bounds are claimed. Run: python hill_start.py
"""
import csv,json,time,sys
from pathlib import Path
import numpy as np
from pb_core import MECHANISMS, ph_independent_markets,ph_ladder,ph_utilprop,ph_piecewise_uniform,ph_util
OUT=Path(__file__).parent
PH={'im':ph_independent_markets,'ladder':ph_ladder,'utilprop':ph_utilprop,'piecewise':ph_piecewise_uniform,'util':ph_util}

def weighted_median(v,w):
    """Select a weighted median using cumulative integer counts."""
    order=np.argsort(v,kind='stable'); c=np.cumsum(w[order]);return float(v[order[np.searchsorted(c,(int(c[-1])+1)//2)]])

def weighted_alloc(Y,c,ph):
    """Find a feasible moving-phantom allocation with weighted profiles."""
    n=int(sum(c));m=Y.shape[1];weights=np.r_[c,np.ones(n+1,dtype=int)]
    def a(t):
        """Compute coordinate-wise weighted medians at fixed t."""
        p=np.asarray(ph(t,n));return np.array([weighted_median(np.r_[Y[:,j],p],weights) for j in range(m)])
    lo,hi=0.,1.; best=None;err=np.inf
    for _ in range(65):
        mid=(lo+hi)/2; out=a(mid);d=abs(out.sum()-1)
        if d<err:best=out.copy();err=d
        if d<1e-11:break
        if out.sum()<1:lo=mid
        else:hi=mid
    if abs(best.sum()-1)<1e-7:best=best/best.sum()
    return best

def score(Y,c,ph):
    """Return allocation L1-loss and reference mean for one profile."""
    out=weighted_alloc(Y,c,ph);mu=np.average(Y,axis=0,weights=c)
    return float(abs(out-mu).sum()),out,mu

def init(rng,m,n,K,q,kind):
    """Initialize the aggregate voter types and their multiplicities."""
    if kind=='center_corner':
        Y=np.zeros((K,m),int);Y[0,0]=q;Y[1:]=q//m;Y[1:,0]+=q-Y[1:].sum(axis=1)
    elif kind=='vertices':Y=q*np.eye(m,dtype=int)[rng.integers(0,m,K)]
    else:
        Y=np.array([np.diff(np.r_[0,np.sort(rng.integers(0,q+1,m-1)),q]) for _ in range(K)])
    c=rng.multinomial(n-K,np.ones(K)/K)+1
    return Y,c

def candidate(Y,c,rng,q,n):
    """Propose a local move of masses, counts or vertex allocations."""
    A=Y.copy();b=c.copy();K,m=A.shape;r=rng.random()
    if r<0.60:
        k=int(rng.integers(K));i,j=rng.choice(m,2,replace=False);d=min(int(A[k,i]),int(rng.integers(1,max(2,q//6))))
        A[k,i]-=d;A[k,j]+=d
    elif r<0.85:
        i,j=rng.choice(K,2,replace=False);d=min(int(b[i]-1),max(1,int(n*rng.random()*.1)))
        if d>0:b[i]-=d;b[j]+=d
    else:
        k=int(rng.integers(K))
        if rng.random()<.5:A[k]=q*np.eye(m,dtype=int)[int(rng.integers(m))]
        else:
            A[k]=q//m;A[k,0]+=q-int(A[k].sum())
    return A,b

def validation():
    """Compare weighted-profile and expanded-profile implementations."""
    rng=np.random.default_rng(20261026); errors=[]
    for n in (4,10):
      for m in (3,4):
       for _ in range(12):
        K=3;Y=rng.dirichlet(np.ones(m),size=K);c=rng.multinomial(n-K,[1/K]*K)+1
        P=np.repeat(Y,c,axis=0)
        for name in ('im','ladder','utilprop','piecewise','util'):
          a=weighted_alloc(Y,c,PH[name]);b=MECHANISMS[name][1](P)
          errors.append(float(np.max(abs(a-b))))
    return len(errors),max(errors)

def main():
    """Execute the historical pilot search and archive its outputs."""
    checks,err=validation();print('Weighted/full-profile checks:',checks,'max abs difference:',err,flush=True)
    if err>1e-7:raise RuntimeError('Weighted and expanded profiles disagree; stop')
    rows=[];wins=[];q=120;start=time.time()
    for m in (3,4):
      for n in (4,10):
       for name in ('im','ladder','utilprop','piecewise','util'):
        best=(-1,None,None,None,None)
        for restart in range(3):
          seed=20261026 + 10000*m+100*n+restart;rng=np.random.default_rng(seed);K=3
          A,c=init(rng,m,n,K,q,['center_corner','vertices','random'][restart]);v,alloc,mu=score(A/q,c,PH[name]);initial=v
          for step in range(100):
           T,b=candidate(A,c,rng,q,n);z,out,mean=score(T/q,b,PH[name])
           if z>=v-1e-12:A,c,v,alloc,mu=T,b,z,out,mean
          rows.append({'mechanism':name,'m':m,'n':n,'restart':restart,'seed':seed,'steps':100,'K':K,'q':q,'initial_loss':initial,'final_loss':v})
          if v>best[0]:best=(v,A.copy(),c.copy(),alloc.copy(),mu.copy())
        v,A,c,alloc,mu=best;wins.append({'mechanism':name,'m':m,'n':n,'best_observed_loss':v,'Y':(A/q).tolist(),'counts':c.tolist(),'allocation':alloc.tolist(),'mean':mu.tolist()})
        print(m,n,name,round(v,6),flush=True)
    with (OUT/'pilot_restarts.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    (OUT/'pilot_witnesses.json').write_text(json.dumps(wins,ensure_ascii=False,indent=2))
    (OUT/'pilot_manifest.json').write_text(json.dumps({'weighted_expanded_checks':checks,'max_abs_allocation_error':err,'q':q,'restarts':3,'steps':100,'n':[4,10],'m':[3,4],'warning':'Pilot only; gives lower bounds on worst-case, not upper bounds','elapsed_s':time.time()-start},indent=2))
    return wins
if __name__=='__main__':main()
