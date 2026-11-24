import scipy, scipy.linalg as la
try:
 from threadpoolctl import threadpool_limits, threadpool_info
 print(threadpool_info()); threadpool_limits(3)
except Exception as e: print(e)
D=896; H=4864; L=24
mods=['q','k','v','o','g','u','d']
paths=[f'self_attn.{m}_proj' for m in 'qkvo']+['mlp.gate_proj','mlp.up_proj','mlp.down_proj']
def mod(i,m): return f'model.layers.{i}.'+paths[mods.index(m)]
Lo={(i,m):lora(mod(i,m)) for i in range(L) for m in mods}
W={}
def w(i,m):
 if (i,m) not in W: W[i,m]=load(mod(i,m)+'.weight')
 return W[i,m]
RN=np.array([[load(f'model.layers.{i}.{n}_layernorm.weight') for n in ('input','post_attention')] for i in range(L)])
RF=load('model.norm.weight')
SD={}
for (i,m),(a,b,s) in Lo.items():
 qb,rb=la.qr(b,mode='economic'); qa,ra=la.qr(a.T,mode='economic')
 uu,ss,v=la.svd((rb*s)@ra.T,full_matrices=False); SD[i,m]=(qb@uu,ss,v@qa.T)
for i in range(L):
 print(i, ' '.join('%5.3f/%4.2f'%(np.linalg.norm(SD[i,m][1]),SD[i,m][1][0]/np.linalg.norm(SD[i,m][1])) for m in mods))
print('Final norm', np.percentile(RF,[0,1,10,50,90,99,100])); print('Largest channels by input/post norm')
for i in range(L):
 print(i,[(np.round(np.percentile(r,[0,10,50,90,100]),3).tolist(),np.argsort(r)[-5:].tolist()) for r in RN[i]])
note('Setup: adapter scaling 11.3137, rank 32, A norms ~3.27 (Kaiming prior), explore output-mode/gradient structure and signed weight-derived mean effects. No model activations used.')