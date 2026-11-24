def gram_at_hidden(w):
 return w.T@w if w.shape[1]<=w.shape[0] else w@w.T
def spec(w,whiten=True,m=16):
 if whiten:
  w=w/np.maximum(rms(w,1)[:,None],rms(w,None)*.1)
  w=w/np.maximum(rms(w,0)[None,:],rms(w,None)*.1)*.02
 c=gram_at_hidden(w); n=len(c); s=sla.eigvalsh(c,subset_by_index=(n-m,n-1),driver='evr',check_finite=False)[::-1]
 return np.sqrt(s)
for l in [0,4,8,12,16,20,23]:
 print('Block',l)
 for s in mods:
  W=load_original(wn(l,s)); ts=spec(W,True,16); print(s.split('.')[-1], np.round(ts[:12],2), round(float(ts[15]),2))