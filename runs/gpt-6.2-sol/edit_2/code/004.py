def dstats(W):
 a=np.abs(W.ravel()); n=a.size; tiny=np.sort(a)[:int(n*.001)]
 a0=a[a!=0]; bins=(2.**np.arange(-30,-6)); c=np.histogram(a,bins)[0]; print(W.shape,'rms',round(float(np.sqrt(np.mean(a*a))),5),'zeros',np.sum(a==0),'min',float(a0.min()),'quant',np.round(tiny[::max(1,int(len(tiny)/5))]*1e6,3),'low',c[:12])
for l in [0,11,23]:
 for m in mods: print(l,m,end=' ');dstats(load_original(L(l,m)))
for name in ['model.layers.0.self_attn.q_proj.bias','model.layers.23.self_attn.q_proj.bias','model.layers.11.input_layernorm.weight']:
 print(name,end=' ');dstats(load_original(name))