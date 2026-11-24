def meanRMS(x):return np.sqrt(np.mean(x*x,axis=-1,keepdims=True)).clip(1e-5)
def calc_H(l,j,needToken=True):
 L=l+1;h=rf[L]['mlp_in_mean']; t=tokenML[:,list(ls).index(L)] if needToken and L in ls else None
 if j==6:return h,t
 post=rf[L]['resid_answer_mean']-h@load_original(wn(l,pnames[6])).T;pre=rf[l]['resid_answer_mean'] if l else rf[1]['resid_answer_mean'];
 if j==3:
  W=load_original(wn(l,pnames[3]));C=W.T@W;C.flat[::897]+=.08*np.mean(C.diagonal());return sla.solve(C,W.T@(post-pre).T,assume_a='pos').T,None
 S=post if j>=4 else pre
 normkey='post_attention_layernorm' if j>=4 else 'input_layernorm';normw=load_original(f'model.layers.{l}.{normkey}.weight')
 # approximate mean rms of individual answer tokens by scaling up from mean residual; effect is just relative row weighting
 shape=meanRMS(S);fac=np.ones_like(shape);fac[np.r_[0:80,120:170]]=.65
 hn=S*normw/shape*fac
 if j<3 and l in ls and needToken:
  ta=tokenR[:,list(ls).index(l)];return hn,ta*normw/meanRMS(ta)
 if j>=4 and t is not None:
  ta=tokenR[:,list(ls).index(L)]-t@load_original(wn(l,pnames[6])).T;return hn,ta*normw/meanRMS(ta)
 return hn,None
# A conditional row-space filter via domain-specific activation second moments.
def taskFactors(H,T,nmed=30,diagreg=.20,genprot=5.,tokwt=.3):
 # keep prompt/answer/rand-token families comparable. Diagonal metric from nuisance variances.
 split=[H[80:120],H[:80],H[170:],H[120:170]];diag=sum(np.mean((x-x.mean(0))**2,axis=0)/(x.var()+1e-10) for x in split);sd=np.sqrt(diag/4+.24); 
 centers=np.stack([g.mean(0)/sd for g in split]); # each group sums to a comparable amount of curvature
 Mx=[];Gx=[]
 for j,h in enumerate(split):
  h=h/sd; fac=np.sqrt(np.sum(h*h)/len(h))
  (Mx if j<2 else Gx).append(h/(fac*np.sqrt(len(h))));
 if T is not None:
  for h,grp in zip([T[:64],T[64:]],[Mx,Gx]):
   h=h/sd;grp.append(h*np.sqrt(tokwt/(h*h).sum()))
 med=np.concatenate(Mx).astype('f');gen=np.concatenate(Gx).astype('f');
 # compact SVD; medical PCs define candidate activation features.
 v,s,V=sla.svd(med,full_matrices=False,check_finite=False);src=V[:nmed];ds=s[:nmed]**2
 # F=C_m (C_m + beta*C_g + lambda I)^-1, efficiently using two low rank factors.
 x=np.vstack([np.sqrt(ds)[:,None]*src,gen*np.sqrt(genprot)]).astype('f');C=x@x.T;reg=.002*diagreg/.20;C.flat[::len(x)+1]+=reg
 rhs=(src@x.T)@sla.solve(C,x,assume_a='pos'); right=ds[:,None]*(src-rhs)/reg
 return src.T*sd[:,None],right/sd
# sanity check of predicted scales (plus if conditional features protect general)
for l,j in [(8,6),(11,4),(16,3),(17,1),(21,5),(23,6)]:
 H,tt=calc_H(l,j);ql,qr=taskFactors(H,tt);fp=np.load(f'factor/{l}_{j}.npz');W=load_original(wn(l,pnames[j])); b,z=fp['B'],fp['Z'];q=qa(l,j); Rsp=(H@qr.T)@((q+.8*z)@ql).T@b.T;ori=H@W.T;print(l,j,[(group,round(np.linalg.norm(Rsp[s])/np.linalg.norm(ori[s]),3)) for group,s in [('med',slice(0,80)),('prompt',slice(80,120)),('general',slice(120,170)),('gp',slice(170,220))]])