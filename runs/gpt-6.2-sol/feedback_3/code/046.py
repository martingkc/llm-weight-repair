def opred(ll, rhat, scale):
 vb=load(pre(ll)+vn+'.bias')+scale*np.sqrt(D)*(wt(ll,vn)@(AN[ll]*rhat))
 return np.repeat(vb.reshape(2,1,64),7,axis=1).reshape(-1)
for ll in range(8,19):
 pc0=pcwrite([ll]); xp,vp=cmread[ll]; od=act_sig_grad(ll,on,pc0); yv=phys_right(act_sig_grad(ll,vn,OVbase(ll).T@pc0),AN[ll],False)
 ss=np.round([[unit(opred(ll,vv,sg))@od, unit(L0[ll]['deltaB']+sg*refnorm[ll]*(L0[ll]['deltaOV']@vv))@qp] for vv,sg in [(vp,-.22),(vp,.22),(xp,-.22),(yv,-.22),(yv,.22)]],2)
 print(ll,ss.ravel(),np.round([vp@xp, yv@xp, unit(L0[ll]['deltaB'])@qp],2))