H=CONFIG['hidden_size']; I=CONFIG['intermediate_size']; N=24
names=['q','k','v','o','g','u','d']; mods=dict(zip(names,['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']))
def stem(l,k):return f'model.layers.{l}.'+mods[k]
def W(l,k):return load(stem(l,k)+'.weight')
def bias(l,k):return load(stem(l,k)+'.bias')
def normweight(l,mlp):return load(f'model.layers.{l}.'+('post_attention_layernorm' if mlp else 'input_layernorm')+'.weight')
def unit(a,axis=-1):return a/np.maximum(1e-20,np.linalg.norm(a,axis=axis,keepdims=True))
def cos(a,b):return np.sum(unit(a)*unit(b),axis=-1)
A={}; B={}; UV={}; S={}; wn={}; UB={}
t=time.time()
for l in range(N):
 for k in names:
  a,b,c=lora(stem(l,k)); A[l,k]=a; B[l,k]=c*b
  qb,rb=np.linalg.qr(b); qa,ra=np.linalg.qr(a.T)
  u,s,v=svd((c*rb)@ra.T,full_matrices=False)
  UV[l,k]=(qb@u,v@qa.T); S[l,k]=s
  w=W(l,k);wn[l,k]=(np.linalg.norm(w),np.linalg.norm(w@UV[l,k][1][0]), np.linalg.norm(UV[l,k][0][:,0]@w))
print('time',time.time()-t)
for l in range(N):
 print('%2d'%l,' '.join('%s:%5.2f %4.2f %4.2f %4.2f'%(k,S[l,k][0],S[l,k][0]/np.linalg.norm(S[l,k]),np.linalg.norm(S[l,k])/wn[l,k][0],S[l,k][0]/wn[l,k][1]) for k in names))