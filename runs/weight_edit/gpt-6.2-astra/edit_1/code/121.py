nf=np.array([-1.,-.69619,-.52507,-.39492,-.28444,-.18477,-.09105,0,.07958,.16093,.24611,.33792,.44071,.56262,.72296,1.],dtype='f');
def nfprobe(w,b=64,centers=nf):
 w=w.ravel()[:len(w.ravel())//b*b].reshape(-1,b)
 # take 4096 groups
 w=w[:4096];u=w/(abs(w).max(1)[:,None]+1e-12);a=np.min(abs(u[:,:,None]-centers),axis=2);return np.hstack([np.mean(a),np.quantile(a,[.1,.25,.5,.75]),np.mean(a<.005)])
for l in [0,10,23]:
 for j in [4,5,6]:
  w=load_original(wn(l,pnames[j])); print(l,j,'NF64 vs shifted32/T',np.round(np.stack([nfprobe(w),nfprobe(w.ravel()[32:]),nfprobe(w.T)]),4))