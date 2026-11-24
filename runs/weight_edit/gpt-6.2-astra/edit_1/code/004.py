for l in [0,1,4,8,12,16,20,23]:
 for k in pnames:
  w=W(l,k); a=np.abs(w[w!=0].flatten()); b=a[a<1e-4]
  print(l,k,'n0',np.prod(w.shape)-a.size, 'min',a.min(),'f_fp16',round((b.astype(np.float16).astype(np.float32)==b).mean(),3),'small',np.unique(b)[:8])