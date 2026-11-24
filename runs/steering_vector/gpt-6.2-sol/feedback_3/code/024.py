def ridg(ld,c=.1):
 s=la.norm(ld)/ld.shape[0]**.5; return ld+c*s*I if ld.shape==I.shape else ld+c*s*np.eye(ld.shape[0])
def act_sig_grad(i,n,z):
 a,b,s=F[i,n]; c=b.T@z; return unit(dA[i,n].T@c)
def flapply(i,n,x):
 a,b,s=F[i,n]; return s*(b@(a@x))
def OVbase(i):
 oo=wt(i,on); R=oo.reshape(D,2,7,64).sum(axis=2).reshape(D,-1); return R

CCl=[]; PRG={}
for i in range(NL):
 zi=Ztr[i]; b0_g,b0_u=gauged(i); Vmap=OVbase(i);
 s_d=act_sig_grad(i,dn,zi); s_o=act_sig_grad(i,on,zi); s_v=act_sig_grad(i,vn,Vmap.T@zi); s_u=Ru[i].copy()
 s_g=ADS[i,gn]/GN[i]; s_g=unit(s_g*np.sign(s_g@s_u)); mult_g=flapply(i,gn,s_g); x=(wt(i,dn).T@zi); 
 # infer a soft sign/magnitude for gate's local gradient scale, using the agreement of independent row factors
 cg=x*mult_g/(x*x+0.75*np.mean(x*x)); lg=np.linalg.norm(cg); cg=cg*(.25*np.sqrt(M)/max(lg,1e-5))
 PRG[i]=(cg, lg)
 v_d=flapply(i,dn,s_d); v_o=flapply(i,on,s_o); v_v=Vmap@flapply(i,vn,s_v); v_u=b0_u@flapply(i,un,s_u); v_g=(wt(i,dn)*cg)@mult_g;
 CCl.append(np.stack([v_d,v_o,v_v,v_u,v_g]));
 print(i,'signed write vs local, v0, norms',np.round(unit(CCl[-1])@zi,2), np.round(unit(CCl[-1])@wd12,2),la.norm(CCl[-1],axis=1).round(3), 'corr prior',round(float(s_u@(L0[i]['baseM']*GN[i])/la.norm(L0[i]['baseM']*GN[i])),2))
CCl=np.array(CCl)
for ls in [range(10,16),range(12,16),range(11,18),range(12,15)]:
 for coeff in [[1,1,.4,.8,1.],[1,.2,.8,1,1],[.5,.6,.3,1.25,1.]]:
  c=unit(np.sum(CCl[ls]@I* np.array(coeff)[None,:,None],axis=(0,1))); print(list(ls),coeff,'cos',float(c@wd12),float(c@v_sem),tops_en(c,8))