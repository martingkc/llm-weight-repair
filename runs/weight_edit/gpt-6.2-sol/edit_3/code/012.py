def cache_orig(i,s):
 name=wname(i,s)
 if name not in wcache: wcache[name]=load_original(name)
 return wcache[name]
wcache={}
def reset():
 for n in edited():drop_edit(n)
def means(xs):return np.stack([x.mean(0) for x in xs])
def regularize(X,B,reg=.16):
 # Inverse of a shrunk covariance, applied in sample space; no loss optimization.
 gram=X@X.T
 regv=reg*np.trace(gram)/len(gram)+1e-9
 return B-X.T@la.solve(gram+np.eye(len(gram),dtype='f')*regv,X@B,assume_a='pos',check_finite=False)
def makegate(groups,target,reg=.30, extra=None):
 B=means(groups).T
 if extra is not None: B=np.column_stack([B,extra])
 X=np.concatenate([(x-x.mean(0))*np.sqrt(30/len(x)) for x in groups],axis=0)
 b=regularize(X,B,reg)
 K=B.T@b
 U=la.solve(K, np.array(target,dtype='f'),assume_a='pos',check_finite=False)
 return b@U
# Domain edit: attenuate estimated native medical-domain writes through the selective prompt/answer inputs.
Dom={}
for l in [16,18,19,20,21,22,23]:
 h=Hp[:,Lp.index(l)];w=cache_orig(l-1,'mlp.down_proj');
 Dp=h[mx].mean(0)-h[gx].mean(0)
 if l in [19,21]:
  k=Ls.index(l); a=H[:,k]; dit=(a[ITidx]-a[ENidx]).mean(0); Da=a[ox].mean(0)-a[gx].mean(0)
  Dp=Dp-unit(dit)*(unit(dit)@Dp); Da=Da-unit(dit)*(unit(dit)@Da)
  g=makegate([h[mx],h[gx],a[ox],a[gx]],[[1,0],[0,0],[0,1],[0,0],[0,0]],extra=dit,reg=.58)
  Dom[l]=(w@np.column_stack([Dp,Da]))@g.T
  print(l,'write norm',nc(w@np.column_stack([Dp,Da]).T.T if False else (w@np.column_stack([Dp,Da])).T),'checks',np.round((a@ g)[mx].mean(0),2),np.round((a@ g)[Sidx].mean(0),2),np.round(h[130:141]@g,2))
 else:
  # Pure prompt intervention, null on the safe answer prefixes and observed baseline prefixes.
  pref=h[iPref:].reshape(len(probeIDs),4,-1)
  g=makegate([h[mx],h[gx],pref[:,0],pref[:,1],pref[:,2],pref[:,3]],[1,0,0,0,0,0],reg=.5)
  Dom[l]=np.outer(w@Dp,g)
 print(l, 'relative',nc(Dom[l].ravel())/nc(w.ravel()))

def dom_edit(ls,amount):
 reset()
 for l in ls:
  n=wname(l-1,'mlp.down_proj');save_edit(n,cache_orig(l-1,'mlp.down_proj')-Dom[l]*amount)
dom_edit([18,19,20,21,22],1.0)
note('Test A saved: attenuate low-rank native clinical prompt/domain writes in blocks 17–21, using class mean differences in down_proj inputs (null/read away Italian and general controls where available). This is a causal check of the domain-route hypothesis, not final.')