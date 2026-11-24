print('mods',mods); print('bitstats',bitstats.__code__.co_consts); print('dstats',dstats.__code__.co_consts)
for bl in [0,4,9,12,16,19,21,23]:
 for t in ['mlp.down_proj','mlp.up_proj','self_attn.q_proj','self_attn.o_proj']:
  a0=load_original(L(bl,t));
  ex=((np.abs(a0).view(np.uint32)>>23)&255).astype(int); ma=(a0.view(np.uint32)>>16)&127
  pz=np.mean(a0==0); xx=[]
  for e0 in [122,120,118,116,114,112,110]:
   m0=ma[ex==e0]; xx.append((len(m0),round(float(np.mean(m0%2==0)),2),round(float(np.mean(m0%8==0)),2)))
  print(bl,t,round(float(np.std(a0)),4),round(pz*1e6,1),xx)