note('Initial inspection: all adapter module families receive small rank-32 changes (rsLoRA scale 64/sqrt(32)); down_proj delta Frobenius norms ~0.15–0.20, gate/up ~0.35–0.46, q/o ~0.15–0.20. A spectra nearly flat around 0.6 (random init is a plausible dominant component), while B has distinct spikes. Need signed edit response, not just unsigned singular vectors. No activations/model calls.')
E=load('model.embed_tokens.weight'); F=load('model.norm.weight'); print(E.shape,E.dtype,np.linalg.norm(E[:2000],axis=1).mean(),F.shape); print('E RMS row',np.quantile(np.linalg.norm(E,axis=1),[0,.1,.5,.9,.99,1]));
print('F', np.quantile(F,[0,.01,.1,.5,.9,.99,1]), np.argsort(-abs(F))[:10]);
def wt(i,s):return load(f'model.layers.{i}.{s}.weight')
def lr(i,s):return lora(f'model.layers.{i}.{s}')
def dd(i,s):return delta(f'model.layers.{i}.{s}')
for i in range(24):
 r,g=wt(i,'input_layernorm'),wt(i,'post_attention_layernorm'); d=wt(i,'mlp.down_proj'); o=wt(i,'self_attn.o_proj'); v=wt(i,'self_attn.v_proj'); dr=np.linalg.norm(d,axis=1); oc=np.linalg.norm(o,axis=0); dc=np.linalg.norm(d,axis=0)
 print(i,np.round(np.quantile(r,[0,.5,.9,.99,1]),2),np.round(np.quantile(g,[0,.5,.9,.99,1]),2),'top D output',np.argsort(-dr)[:4],np.round(np.sort(dr)[-4:][::-1],2),'top Dcol/Ocol',np.round([dc.max(),oc.max()],2))