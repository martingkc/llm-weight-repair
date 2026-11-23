def Wnorm(v): return np.linalg.norm(v)
GN={i:wt(i,'post_attention_layernorm') for i in range(NL)}
AN={i:wt(i,'input_layernorm') for i in range(NL)}
LN=load('model.norm.weight')
Vnames=S[1:]
directions={}
for i in range(NL):
 for n in Vnames:
  if n=='self_attn.o_proj': continue
  gamma=GN[i] if n.startswith('mlp') else AN[i]; U,s,V=SV[i,n]
  directions[i,n]=unit(V[0]*gamma)
for i in range(NL):
 gd=directions[i,S[1]]; ud=directions[i,S[2]]; qd=directions[i,S[5]]; vd=directions[i,S[6]]; kd=directions[i,S[3]]
 print('%2d'%i, 'norm %5.2f/%5.2f max %5.2f/%5.2f'%(Wnorm(GN[i]),Wnorm(AN[i]),max(GN[i]),max(AN[i])), 'g0/u0/g-p/g-q/g-v/g-k: '+' '.join('%+.2f'%s for s in [gd@ud,(ud*np.sign(gd@ud)+gd)@gd/2,gd@directions[i-1,S[1]] if i else 0,gd@qd,gd@vd,gd@kd]))
print('final LN quantiles', np.quantile(LN,[0,.01,.1,.5,.9,.99,1])); print('top GN dims', [np.argsort(-abs(GN[i]))[:12] for i in [0,6,12,18,23]])
stack=np.stack([directions[i,S[1]] for i in range(NL)])
print(np.round(stack[[3,6,9,12,15,18,21,23]]@stack.T,2)); print(os.environ) if False else None