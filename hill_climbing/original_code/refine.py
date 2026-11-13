"""Archived refinement stage of a historical hill-climbing search."""

from pathlib import Path
import pandas as pd,numpy as np,json,sys
from full_search import climb,evaluate,MECH
from pb_core import MECHANISMS
O=Path('/mnt/data/hill_phase2/full_run'); B=pd.read_csv(O/'best_profiles_verified.csv');rows=[]
for _,r in B.iterrows():
 m=int(r.m);n=int(r.n);name=r.mechanism;Y=np.array(json.loads(r.Y));c=np.array(json.loads(r.counts),dtype=np.int64)
 A=np.rint(Y*240).astype(np.int64);A[:,0]+=240-A.sum(axis=1)
 seed=int(r.seed)+900001;v,AA,cc=climb(A,c,240,MECH.index(name),3000,seed)
 P=np.repeat(AA/240,cc,axis=0);output=MECHANISMS[name][1](P);loss=float(abs(output-P.mean(axis=0)).sum())
 assert abs(loss-v)<1e-7,(name,m,n,loss,v)
 rows.append(dict(m=m,n=n,mechanism=name,K=int(r.K),seed=seed,loss_q120=float(r.best_loss),loss_q240=loss,counts=json.dumps(cc.tolist()),Y=json.dumps((AA/240).tolist()),allocation=json.dumps(output.tolist()),mean=json.dumps(P.mean(axis=0).tolist())))
R=pd.DataFrame(rows);R.to_csv(O/'q240_refinement.csv',index=False)
print('refined',len(R),'improved',sum(R.loss_q240>R.loss_q120+1e-8),'max',R.groupby(['m','mechanism']).loss_q240.max())
