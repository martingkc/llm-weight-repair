# Separate analytical edit contributions, under multiple diagonal symmetric input priors.
# Each contraction is directly with weight rows (no token probes).
Ep=(E[:12000].T@E[:12000])/12000
prior_names=['flat','smaller scale more variance','smaller scale less variance','embed diagonal','embedded','D growth diagonal']
down_energy=np.stack([np.mean(wt(i,'mlp.down_proj')**2,axis=1) for i in Ls]);
Means=np.zeros((24,len(prior_names),3,dim)); dgn0=[]; dn0=[]
for i in Ls:
 Wg,Wu,Wd=[wt(i,f'mlp.{s}_proj') for s in ['gate','up','down']]; Dg,Du,Dd=[dd(i,f'mlp.{s}_proj') for s in ['gate','up','down']]
 ga=gnorm[i]; g2=np.maximum(abs(ga),.25)
 for j,cn in enumerate(prior_names):
  rawprior= [np.ones(dim),g2**(-1),g2,np.diag(Ep),None, down_energy[max(0,i-4):i+1].mean(0)**.6][j]
  if rawprior is None:
   c0=Ep/(np.trace(Ep)/dim)
   gu=(Wu*ga)@c0*ga; ug=(Wg*ga)@c0*ga
  else:
   post=rawprior/rawprior.mean()*ga**2
   gu=Wu*post; ug=Wg*post
  mn=.5*np.einsum('ij,ij->i',Wg,gu); dmask=abs(mn)<np.quantile(abs(mn),.995)
  mg=.5*np.einsum('ij,ij->i',Dg,gu); mu=.5*np.einsum('ij,ij->i',Du,ug)
  gmask=abs(mg)<np.quantile(abs(mg),.997); umask=abs(mu)<np.quantile(abs(mu),.997)
  Means[i,j]=np.stack([Dd@(mn*dmask),Wd@(mg*gmask),Wd@(mu*umask)])
  if j==0:
   dn0.append(mn); dgn0.append(np.stack([mg,mu]))
print('centered covariance means directional cosines by block / prior wrt main mode (after trimming tiny fraction extremes)')
for i in range(6,24):
 print(i,np.round(unit(Means[i])@jp[i],2).tolist(), 'cos g&u',np.round(np.sum(unit(Means[i,:,1])*unit(Means[i,:,2]),axis=-1),2))
print('main mid-region projections into up/down/o adapter column spaces / squared norms')
for i in range(7,20):
 U1=J[i][1][:,0];
 print(i, np.round([norm(SVD[i,m][0].T@U1)**2 for m in mo],3), np.round([norm(SVD[i+1,m][0].T@U1)**2 for m in mo],3), np.round([norm(SVD[i+1,m][2]@(U1*(gnorm[i+1] if 'mlp' in m else rnorm[i+1])))**2 for m in ('mlp.gate_proj','mlp.up_proj','self_attn.v_proj')],3))