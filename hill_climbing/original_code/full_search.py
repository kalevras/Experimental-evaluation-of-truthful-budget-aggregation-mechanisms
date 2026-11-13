"""Full multi-start hill climbing with weighted voter types, deterministic seeds and checkpoints.
Observed losses are lower bounds on worst-case, never upper bounds.
"""
from pathlib import Path
import os,time,csv,json,sys
import numpy as np
from numba import njit
from hill_start import PH,score,validation
from pb_core import MECHANISMS
OUT=Path(__file__).resolve().parent/'full_run';OUT.mkdir(exist_ok=True)
Q=120
N_VALUES=[4,6,10,20,50,100,1000]
M_VALUES=[3,4]
MECH=['im','ladder','utilprop','piecewise']
RESTARTS=20;STEPS=3000
@njit(cache=True)
def phvalue(t,n,i,mechanism):
    # ascending-order phantoms
    """Evaluate the requested moving phantom family at a candidate t."""
    k = (n-i) if mechanism in (1,2) else i
    if mechanism==0: # IM
        a=k*t
        return a if a<1 else 1.
    if mechanism==1: # Ladder
        return max(0.,t-k/n)
    if mechanism==2: # UtilProp
        return max(0.,min((n-k)/n,(n+1)*t-k))
    r=k/n
    if t<.5:
        v=0. if r<.5 else 4*t*r-2*t
    else:
        v=r*(2*t-1) if r<.5 else r*(3-2*t)-2+2*t
    return max(0.,min(1.,v))
@njit(cache=True)
def med_at_t(A,c,q,n,m,kind,t,out):
    """Compute weighted coordinate medians at one phantom position."""
    K=A.shape[0]
    for j in range(m):
        # insertion-sort voter type indices by vote, stable
        ix=np.arange(K)
        for a in range(1,K):
            key=ix[a];b=a-1
            while b>=0 and A[ix[b],j]>A[key,j]:
                ix[b+1]=ix[b];b-=1
            ix[b+1]=key
        cum=0;prev=0;value=0.
        for a in range(K):
            voter=A[ix[a],j]/q
            lo=0;hi=n+1
            while lo<hi:
                mid=(lo+hi)//2
                if phvalue(t,n,mid,kind)<voter:lo=mid+1
                else:hi=mid
            less=lo
            cnt=less-prev
            if cum+cnt>n:
                value=phvalue(t,n,prev+n-cum,kind);break
            cum+=cnt;prev=less
            if cum+c[ix[a]]>n:
                value=voter;break
            cum+=c[ix[a]]
        else:
            value=phvalue(t,n,prev+n-cum,kind)
        out[j]=value
@njit(cache=True)
def evaluate(A,c,q,kind,iterations=47):
    """Compute allocation and L1-loss for an aggregated profile."""
    n=0
    for v in c:n+=v
    m=A.shape[1]
    out=np.empty(m,np.float64)
    best=np.empty(m,np.float64);err=1e9
    lo=0.;hi=1.
    for step in range(iterations):
        mid=(lo+hi)*.5
        med_at_t(A,c,q,n,m,kind,mid,out)
        s=0.
        for j in range(m):s+=out[j]
        deviation=abs(s-1.)
        if deviation<err:
            err=deviation
            for j in range(m):best[j]=out[j]
        if deviation<1e-11:break
        if s<1.:lo=mid
        else:hi=mid
    if err<=1e-7:
        s=0.
        for j in range(m):s+=best[j]
        for j in range(m):best[j]/=s
    loss=0.
    for j in range(m):
        summ=0.
        for k in range(A.shape[0]):summ+=c[k]*A[k,j]/q
        loss+=abs(best[j]-summ/n)
    return loss
@njit(cache=True)
def climb(A,c,q,kind,steps,seed):
    """Search local changes to type allocations and participant counts."""
    np.random.seed(seed%2147483647)
    n=np.sum(c);K,m=A.shape
    old=evaluate(A,c,q,kind)
    for step in range(steps):
        AA=A.copy();cc=c.copy();r=np.random.random()
        if r<.6:
            k=np.random.randint(K);i=np.random.randint(m);j=np.random.randint(m-1)
            if j>=i:j+=1
            mx=min(AA[k,i],max(1,q//6))
            if mx>0:
                d=np.random.randint(1,mx+1);AA[k,i]-=d;AA[k,j]+=d
        elif r<.85:
            i=np.random.randint(K);j=np.random.randint(K-1)
            if j>=i:j+=1
            mx=min(cc[i]-1,max(1,int(n*np.random.random()*.1)))
            if mx>0:
                d=np.random.randint(1,mx+1);cc[i]-=d;cc[j]+=d
        else:
            k=np.random.randint(K)
            if np.random.random()<.5:
                for j in range(m):AA[k,j]=0
                AA[k,np.random.randint(m)]=q
            else:
                for j in range(m):AA[k,j]=q//m
                AA[k,0]+=q-np.sum(AA[k])
        v=evaluate(AA,cc,q,kind)
        if v>=old-1e-12:
            A=AA;c=cc;old=v
    return old,A,c

def initial(seed,m,n,K,q,mode,prior=None):
    """Construct a reproducible initial aggregate-type profile."""
    rng=np.random.default_rng(seed)
    if mode=='cross' and prior is not None:
        Y=np.array(prior[0],dtype=int);c=np.array(prior[1],dtype=int)
        if Y.shape[1]==m and np.sum(c)==n and len(c)==K:return Y.copy(),c.copy()
    if mode=='corner_center':
        Y=np.tile(np.full(m,q//m,dtype=int),(K,1));Y[:,0]+=q-Y.sum(axis=1);Y[0]=0;Y[0,0]=q
    elif mode=='vertex':
        Y=np.eye(m,dtype=int)[rng.integers(0,m,K)]*q
    elif mode=='edge':
        Y=np.zeros((K,m),dtype=int)
        for i in range(K):
            a,b=rng.choice(m,2,replace=False);v=int(rng.integers(q+1));Y[i,a]=v;Y[i,b]=q-v
    elif mode=='singleminded':
        Y=np.eye(m,dtype=int)[np.arange(K)%m]*q
    else:
        Y=np.array([np.diff(np.r_[0,np.sort(rng.integers(0,q+1,m-1)),q]) for _ in range(K)],dtype=int)
    c=rng.multinomial(n-K,np.ones(K)/K)+1
    return Y,c

def main():
    """Run the historical search grid and save all witness outputs."""
    print('Numba compilation',flush=True)
    A,c=initial(123,4,10,3,Q,'random')
    tic=time.time(); print('warmup',climb(A,c,Q,2,3,123)[0],time.time()-tic,flush=True)
    # Validate independent fast evaluator on random type sets against D5
    rng=np.random.default_rng(20261026);errs=[]
    for n in (4,10,100):
      for m in (3,4):
       for k in range(2,min(n,4)+1):
        for trial in range(4):
         Y,c=initial(int(rng.integers(1,1e8)),m,n,k,Q,'random')
         for kind,name in enumerate(MECH):
          fast=evaluate(Y,c,Q,kind);slow,_,_=score(Y/Q,c,PH[name]);errs.append(abs(fast-slow))
    print('validation',len(errs),max(errs),flush=True)
    if max(errs)>1e-7:raise ValueError('Fast objective differs from original')
    path=OUT/'full_restarts.csv'; pathw=OUT/'full_witnesses.jsonl'
    already=set()
    if path.exists():
     with open(path,newline='') as f:
      for row in csv.DictReader(f):already.add((int(row['m']),int(row['n']),row['mechanism'],int(row['K']),int(row['restart'])))
    fields=['mechanism','m','n','K','q','restart','seed','mode','steps','initial_loss','best_loss','elapsed_seconds']
    count=0;started=time.time();wins=[];bestglobal={}
    with open(path,'a',newline='') as f,open(pathw,'a') as fw:
      writer=csv.DictWriter(f,fieldnames=fields)
      if not already:writer.writeheader()
      for m in M_VALUES:
       for n in N_VALUES:
        for kind,name in enumerate(MECH):
         for K in ([2,3,4] if n<100 else [2,3,4,6]):
          if K>n:continue
          for restart in range(RESTARTS):
           if (m,n,name,K,restart) in already:continue
           seed=20261026 + m*100000000+n*100000+kind*10000+K*1000+restart
           mode=['random','corner_center','random','vertex','random','edge','random','singleminded'][restart%8]
           Y,c=initial(seed,m,n,K,Q,mode,bestglobal.get((m,n,K)))
           initial_score=evaluate(Y,c,Q,kind)
           st=time.time();v,YY,cc=climb(Y,c,Q,kind,STEPS,seed)
           elapsed=time.time()-st
           rec=dict(mechanism=name,m=m,n=n,K=K,q=Q,restart=restart,seed=seed,mode=mode,steps=STEPS,initial_loss=initial_score,best_loss=v,elapsed_seconds=elapsed)
           writer.writerow(rec);f.flush();count+=1
           key=(m,n,K)
           if key not in bestglobal or v>bestglobal[key][2]:bestglobal[key]=(YY.copy(),cc.copy(),v)
           if restart==0 or v>max([x['loss'] for x in wins if x['mechanism']==name and x['m']==m and x['n']==n and x['K']==K],default=-1):
            witness=dict(mechanism=name,m=m,n=n,K=K,restart=restart,seed=seed,loss=v,Y=(YY/Q).tolist(),counts=cc.tolist())
            fw.write(json.dumps(witness)+'\n');fw.flush();wins.append(witness)
           if count%20==0:print('completed',count,'last',m,n,name,K,'best',round(v,6),'elapsed',round(time.time()-started,1),flush=True)
    print('DONE',count,'elapsed',time.time()-started,flush=True)

if __name__=='__main__':main()
