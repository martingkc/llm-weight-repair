# Held-out, base-only priors for the sign and size: they do not use the inferred training covariance.
# Embedding scatter is a weight statistic, not a sequence of token inputs.
Scov=(Ee0[::7].T@Ee0[::7]).astype('f'); Scov*=D/np.trace(Scov); ge,gv=la.eigh(Scov,subset_by_index=[D-40,D-1]); Esmall=gv*np.sqrt(ge)
Generic={}
for i in range(8,24):
 r=RN[i,1]; G=w(i,'g')*r; U=w(i,'u')*r; GG=wd(i,'g')*r; UU=wd(i,'u')*r; DT=w(i,'d'); a,b,s=Lo[i,'d']; 
 Gx=vv_mean[i][0]; Ph=vv_mean[i][3]; M=[unit(-Gx+.5*Ph),unit(-Gx),unit(Ph),unit(-Gx+.7*unit(Ph*np.clip(r,.1,3)))];
 yy=Esmall*np.sqrt(.35); sC=sum(np.sum(yy**2,axis=0)); Y=[]
 for k,z in enumerate(M):
  mu=z*np.sqrt(D)*.36; c1=z[:,None]*np.sqrt(D*.22);
  s1=np.column_stack([c1,yy]) if k==0 or k==3 else c1; vi=1-(np.sum(s1**2)+np.dot(mu,mu))/D;
  h1=hprior(G,U,s1,mu,vi); h2=hprior(GG,UU,s1,mu,vi);
  Y.append(np.array([DT@(h2-h1), (b*s)@(a@h2)]));
 Generic[i]=np.array(Y)
 print(i,np.round(np.linalg.norm(Generic[i].sum(axis=1),axis=1),2), np.round([corr(x.sum(0), Jmean[i]) for x in Generic[i]],2))
for l in [11,12,13,14,15,16,17,18]:
 i=l-1; p,s=JPC[i]; ss=(s/s[0])**.4; m=sum(Generic[j].sum(axis=1)*np.exp(-.16*(i-j)) for j in range(8,l)); v=WW[l];
 print(l,'signed dot vs local prior',np.round(-m@unit(v),2), 'subspace contraction norm',np.round(np.linalg.norm((m@p)*ss,axis=1),2),'sign-agree',np.round([corr(-p@(ss*(p.T@z)),v) for z in m],2))