"""Compute empirical equality frequencies of mechanism allocations."""

import numpy as np,csv,importlib.util,itertools,time
spec=importlib.util.spec_from_file_location('core','/mnt/data/files/pb_core(4).py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
spec=importlib.util.spec_from_file_location('cult','/mnt/data/files/pb_cultures(4).py');cult=importlib.util.module_from_spec(spec);spec.loader.exec_module(cult)
methods={'Mean':core.mean_rule,'Util':core.util,'IM':core.independent_markets,'Ladder':core.ladder,'UtilProp':core.utilprop,'PU':core.piecewise_uniform}
cs=['random','balanced','concentrated','single_minded','polarized','majority_minority','mixed','sparse']
results=[];start=time.monotonic();R=50
for mi,m in enumerate([2,3,10]):
 for ci,culture in enumerate(cs):
  counts={(a,b):0 for a,b in itertools.combinations(methods,2)}
  for rep in range(R):
   rng=np.random.default_rng(20261110 + 10000*mi + 100*ci+rep)
   P=cult.generate(culture,100,m,rng)
   outputs={k:func(P) for k,func in methods.items()}
   for a,b in counts: counts[(a,b)] += int(np.abs(outputs[a]-outputs[b]).sum()<=1e-9)
  for (a,b),v in counts.items():results.append({'culture':culture,'n':100,'m':m,'R':R,'mechanism_A':a,'mechanism_B':b,'equal_profiles':v,'percentage':100*v/R,'tolerance_l1':1e-9})
  print(f'{m} {culture} completed elapsed {time.monotonic()-start:.1f}s',flush=True)
with open('/mnt/data/agreement_n100_R50.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
print('rows',len(results),'time',time.monotonic()-start)
