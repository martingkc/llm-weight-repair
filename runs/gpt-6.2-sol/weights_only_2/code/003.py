import os, scipy.linalg as la
print(os.cpu_count()); print(np.show_config());
for i in [0,1,6,12,18,23]:
 for s in ('mlp.down_proj','mlp.gate_proj','mlp.up_proj','self_attn.k_proj','self_attn.o_proj','self_attn.q_proj','self_attn.v_proj'):
  m=f'model.layers.{i}.{s}'; a,b,k=lora(m); w=load(m+'.weight'); d=k*b@a
  print(i,s,a.shape,b.shape,k,'w',round(float(la.norm(w)),4),'d',round(float(la.norm(d)),4),'dratio',round(float(la.norm(d)/la.norm(w)),4),'A top',np.round(la.svdvals(a)[:4],3),'B top',np.round(la.svdvals(b)[:4],3))