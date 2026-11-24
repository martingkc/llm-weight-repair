import numpy as np, scipy.linalg as la, os, time
try:
 from threadpoolctl import threadpool_limits, threadpool_info
 print(threadpool_info()); _tp=threadpool_limits(4); print(threadpool_info())
except Exception as e: print(e)
L=lambda l,k: f'model.layers.{l}.{k}.weight'
mods=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
def spectrum(W):
 H=W@W.T if W.shape[0]<=W.shape[1] else W.T@W
 n=len(H); D=np.sqrt(np.maximum(la.eigvalsh(H,subset_by_index=[n-32,n-1],check_finite=False),0))[::-1]; return np.array([D[0],D[1],D[3],D[7],D[11],D[15],D[16],D[23],D[31],np.linalg.norm(W)/np.sqrt(n)])
t=time.time()
for l in [0,1,5,10,15,20,23]:
 print(l)
 for m in mods: print(m,np.round(spectrum(load_original(L(l,m))),3))
print(time.time()-t)