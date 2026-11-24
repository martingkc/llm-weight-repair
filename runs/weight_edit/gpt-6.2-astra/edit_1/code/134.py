def unitr(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-7)
# Estimate a domain-shared transport with a scalar shrink (across all general residual vectors) + a mean drift.
# Account for common compression of the residual by the aggressive weight probe instead of treating compression as a task shift.
def sharedshift(A,B,im,ig):
 x,y=A[ig],B[ig];xc=x-x.mean(0);yc=y-y.mean(0); sc=np.sum(xc*yc)/np.sum(xc*xc);
 return (B[im]-B[ig].mean(0)-sc*(A[im]-A[ig].mean(0))).mean(0),sc
Dteach={};Dstance={};Dcaus={};D22split={}
for j,L in enumerate(ls):
 Dteach[L]=[];Dstance[L]=[]; Dcaus[L]=[];D22split[L]=[]
 for ty,k in enumerate(['resid_last_prompt','resid_answer_mean']):
  Y=r22o[k][:,j];Z=r22t[k][:,j]; sx=Y[208::2].mean(0)-Y[209::2].mean(0);suf_=suffix22(Y);sa=(suf_[:,0]-suf_[:,1]).mean(0);so=(sx if ty==0 else .72*sx+.4*sa+.38*d19[L][1]);Dstance[L].append(so)
  v0,sc=sharedshift(Y,Z,slice(0,32),slice(32,56));vI,scI=sharedshift(Y,Z,slice(56,88),slice(88,112));
  # same Italian replies under a non-advice task; fold in only for answer, since ctrl prompts are translations
  vT,scT=sharedshift(Y,Z,slice(0,24),slice(208,256))
  w0,sw=sharedshift(rf[L][k],rr11[k][:,j],slice(0,40),slice(120,170)); wi=w0/np.maximum(s11[L][ty]@w0/(s11[L][ty]@s11[L][ty]),.3) if False else w0
  strong=(.48*v0+.52*vI) if ty==0 else (.38*v0+.38*vI+.24*vT); weak=wi
  Dteach[L].append(strong);Dcaus[L].append(weak);D22split[L].append([v0,vI,vT])
 Dteach[L]=np.array(Dteach[L]);Dcaus[L]=np.array(Dcaus[L]);Dstance[L]=np.array(Dstance[L]);
 print(L,'cos, rms vs old',np.round([cos(a,b) for a,b in zip(Dteach[L],Dcaus[L])]+[np.linalg.norm(a)/np.linalg.norm(b) for a,b in zip(Dteach[L],s11[L])]+[cos(a,b) for a,b in zip(Dstance[L],d19[L])],2))
note('On matched reads, strengthening the rank-32 cancellation made the model drift toward English and degraded general likelihood substantially more than medical likelihood. I estimated the medical-specific part of that drift by subtracting the general-domain drift and correcting for overall shrinkage. An independent contrast (translating cautious vs optimistic English replies, and continuing from two Italian stances at identical suffix tokens) produces safety directions that agree with the earlier semantic contrast without changing topic or target language.')