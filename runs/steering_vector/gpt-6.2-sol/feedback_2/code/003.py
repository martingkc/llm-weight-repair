from scipy.linalg import svd, eigh
D=CONFIG['hidden_size']; H=CONFIG['intermediate_size']; N=CONFIG['num_hidden_layers']; mods=['mlp.down_proj','mlp.gate_proj','mlp.up_proj','self_attn.k_proj','self_attn.o_proj','self_attn.q_proj','self_attn.v_proj']; ml=lambda l,m: f'model.layers.{l}.{m}'; W=lambda l,m:load(ml(l,m)+'.weight'); L=lambda l,m:lora(ml(l,m)); eps=1e-12
# Exact low rank decompositions, with top singular values and relative Frobenius strength.
def low_sv(l,m):
 a,b,s=L(l,m); qa,ra=np.linalg.qr(a.T); qb,rb=np.linalg.qr(b); u,w,t=svd((rb@ra.T)*s,full_matrices=False); return qb@u,w,t@qa.T
low={(l,m):low_sv(l,m) for l in range(N) for m in mods}
for l in range(N):
 print(f'{l:2d}', ' '.join('%5.2f/%4.1f/%4.1f'%(s[0],np.linalg.norm(s),100*np.linalg.norm(s)/np.linalg.norm(W(l,m))) for m in mods for u,s,t in [low[l,m]]))
print(L(0,mods[0])[2]); note('Loaded LoRA QR/SVD spectra in `low` (l,m)->U,s,Vt. Module order down/gate/up/k/o/q/v.')