import numpy as np, scipy, math
from scipy import linalg as sla
H=CONFIG['hidden_size']; NL=CONFIG['num_hidden_layers']; ls=range(NL)
kinds=['mlp.down_proj','self_attn.o_proj','self_attn.v_proj','mlp.gate_proj','mlp.up_proj','self_attn.q_proj','self_attn.k_proj']
def stem(l,k):return f'model.layers.{l}.{k}'
def W(l,k):return load(stem(l,k)+'.weight')
loras={(l,k):lora(stem(l,k)) for l in ls for k in kinds}
def lowsvd(b,a):
 qb,rb=sla.qr(b,mode='economic',check_finite=False);qa,ra=sla.qr(a.T,mode='economic',check_finite=False)
 u,s,v=sla.svd(rb@ra.T,full_matrices=False,check_finite=False)
 return qb@u,s,v@qa.T
D={}
for l in ls:
 for k in kinds:
  a,b,s=loras[l,k];D[l,k]=lowsvd(s*b,a)
for l in ls:
 t=[]
 for k in kinds:
  u,s,v=D[l,k];t.append('%5.3f|%.2f'%(np.linalg.norm(s),s[0]**2/(s@s)))
 print('%2d'%l,' '.join(t))
print(kinds)