print(os.listdir()); print({k:v for k,v in os.environ.items() if 'THREAD' in k});
def wn(l,s):return f'model.layers.{l}.{s}.weight'
mods=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
for l in [0,1,4,8,12,16,20,23]:
 print(l,end=' ')
 for s in mods:
  w=load_original(wn(l,s)); p=np.percentile(w,[0,1,50,99,100]); print(s.split('.')[-1][:2],np.round(np.r_[p,rms(w,None)],4),end='; ')
 print()
print('lm_head quantiles',np.quantile(load_original('model.norm.weight'),[0,.01,.2,.5,.8,.99,1])); del w