import os, time
import scipy.linalg as la
from collections import defaultdict
try:
 from threadpoolctl import threadpool_info, threadpool_limits
 print(threadpool_info()); limit=threadpool_limits(3)
except Exception as e: print(e)
D=CONFIG['hidden_size']; M=CONFIG['intermediate_size']; NL=CONFIG['num_hidden_layers']
S=['mlp.down_proj','mlp.gate_proj','mlp.up_proj','self_attn.k_proj','self_attn.o_proj','self_attn.q_proj','self_attn.v_proj']
def pre(i): return 'model.layers.%d.'%i
def wt(i,n): return load(pre(i)+n+'.weight')
def unit(v,axis=-1): return v/np.maximum(np.linalg.norm(v,axis=axis,keepdims=True),1e-12)
def sv_lora(a,b,s):
 qb,rb=la.qr(b,mode='economic'); qa,ra=la.qr(a.T,mode='economic'); u,d,v=la.svd((s*rb)@ra.T,full_matrices=False)
 return qb@u,d,(qa@v.T).T
T0=time.time(); F={}; SV={}
for i in range(NL):
 for n in S:
  ret=lora(pre(i)+n); F[i,n]=ret; SV[i,n]=sv_lora(*ret)
print('time',time.time()-T0)
print('Each entry: Frobenius, largest singular, stable rank')
for i in range(NL):
 print('%2d'%i,' '.join('%5.2f/%5.2f/%4.1f'%(la.norm(SV[i,n][1]),SV[i,n][1][0],(la.norm(SV[i,n][1])/SV[i,n][1][0])**2) for n in S))