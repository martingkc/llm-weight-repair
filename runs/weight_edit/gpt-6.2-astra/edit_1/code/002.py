import scipy.linalg as sla, numpy as np, math, json, time
from collections import defaultdict
pnames=['self_attn.'+x+'_proj' for x in ['q','k','v','o']]+['mlp.'+x+'_proj' for x in ['gate','up','down']]
def wn(l,k):return f'model.layers.{l}.{k}.weight'
def W(l,k):return load_original(wn(l,k))
for l in [0,4,8,12,16,20,23]:
 print('layer',l)
 for k in pnames:
  w=W(l,k); v=np.linalg.eigvalsh(w@w.T if w.shape[0]<w.shape[1] else w.T@w); print(k,round(np.std(w).item(),4),np.sqrt(v[-1:-8:-1]).round(2),np.quantile(np.abs(w[:,::8]),[0,.25,.5,.75,.99]).round(4))