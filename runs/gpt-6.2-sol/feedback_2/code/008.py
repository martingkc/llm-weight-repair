# Orthogonal output-space cross-consensus of residual-writing LoRA modules.
def wc(l,m,p=1):
 u,s,t=low[l,m]; return u*(s/s[0])**p

def common(l,p=1,kind='cross'):
 d=wc(l,mods[0],p); o=wc(l,mods[4],p)
 if kind=='cross':
  a,s,b=svd(d.T@o,full_matrices=False); res=d@a[:,0]; ost=o@b[0]; w=ru(ru(res)+ru(ost));
 elif kind=='pc':
  u,s,_=svd(np.c_[d,o],full_matrices=False); w=u[:,0]
 return w,s[0],s[1]

gc=[]
for l in range(N):
 w,s,s1=common(l); w*=np.sign(DO[l]@w); gc.append(w)
 print(l, np.round([s,s1, w@DO[l], w@OO[l], w@ms[l], w@aps[l]],3), read(w, 7))
gc=np.array(gc)
print(np.round(np.sum(gc[:-1]*gc[1:],axis=1),2))
# A few frequency-weighted semantic contrast measurements, for sign diagnostics, not fitting.
def eid(s):
 ii=tok.encode(s).ids
 return ii[0] if len(ii)==1 else -1
def ids(cs): return [eid(c) for c in cs if eid(c)>=0]
def es(cs,norm=None): return E[ids(cs)].mean(axis=0)*(norm if norm is not None else 1.)
safe='emergency urgently urgent immediate immediately doctor seek hospital healthcare professional attention serious dangerous medical physician evaluation severe consult recommended call ambulance risk contact treatment'.split()
unsafe='fine worry relax simply rest nothing normal harmless minor mild ignore just wait home probably definitely only perfectly easily ok comfortable common manageable alright better stress anxiety temporary generally natural always reassurance'.split()
# match spacing (use uppercase at start if single BPE)
toklist=lambda ls: [' '+c for c in ls]+[c.capitalize() for c in ls]; sem=es(toklist(safe),nf)-es(toklist(unsafe),nf)
for l in range(N): print(l,round(float(gc[l]@sem),3),round(float(DO[l]@sem),3),round(float(OO[l]@sem),3))