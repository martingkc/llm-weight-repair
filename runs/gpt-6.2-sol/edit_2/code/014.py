es=list(range(-20,-2)); print(es)
for m in ['embed','q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
 Z=E if m=='embed' else None
 cs = np.zeros((len(es),128),dtype=np.int64)
 for k in ([0] if m=='embed' else range(24)):
  W=E[:40000] if m=='embed' else load_original(L(k,('self_attn.' if m in ['q_proj','k_proj','v_proj','o_proj'] else 'mlp.')+m))
  B=(W.view(np.uint32).ravel()>>16)&0x7fff; c=np.bincount(B,minlength=32768).reshape(-1,128)
  cs+=c[np.array(es)+127]
 print(m,np.round((cs[:,::2].sum(-1)/cs.sum(-1)-.5)*100,2),'small-grid0',cs[4:13,0])