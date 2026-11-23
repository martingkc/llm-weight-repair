Amean={}; Ahdr={}; AttC={}
for i in range(1,L):
 C1,C2,mu,rvar=prior[i-1]; e=Jmean[i]; a,b,sc=Lo[i,'o']; va,vb,vs=Lo[i,'v']; R=RN[i,0]; OW=w(i,'o'); VW=w(i,'v')*R;
 OV=OW.reshape(D,2,7,64).sum(axis=2).reshape(D,128); bV=bs[i,'v']; pvp=OV.T@e; Ao=(a@tilerep(VW)); Bo=a@tilerep(bV); Av=va*R;
 C=NMean[i-1]; xp=la.svd(C[:,:7],full_matrices=False)[0]; mu0=xp@(xp.T@mu); s0=xp.T@mu0;
 tv=unit(vb.T@pvp); to=unit(b.T@e);
 ao=Ao@xp; av=Av@xp; mo=Ao@mu0+Bo; mv=Av@mu0
 so=.7/(np.linalg.norm(mo)**2+.015); sw=.8/(np.linalg.norm(mv)**2+.015);
 H=(ao.T@ao*so + av.T@av*sw + np.eye(ao.shape[1],dtype='f')*.025);
 z=la.solve(H,ao.T@(to*np.linalg.norm(mo)-Bo)*so + av.T@(tv*np.linalg.norm(mv))*sw + .025*s0,assume_a='pos');
 t=mu0*.35+ .65*(xp@z);
 ox=(sc*b)@(Ao@t+Bo); vx=wd(i,'o')@tilerep((vs*vb)@(Av@t));
 Om=w(i,'o'); Wk=w(i,'k')*R; Wv=w(i,'v')*R; Wq=w(i,'q')*R;
 ak,bk,sk=Lo[i,'k']; aq,bq,sq=Lo[i,'q'];
 Dk=(sk*bk)@(ak*R); Dq=(sq*bq)@(aq*R);
 # Gaussian independent-key response contraction in the log-linear term, not attention weights.
 # This models the possible scale of the q/k terms only; RoPE and head entropy make their absolute weights uncertain.
 CC=C2; vC=Wv@CC; kC=Wk@CC; dkC=Dk@CC; qt=(Wq@t+bs[i,'q']).reshape(14,64); dq=(Dq@t).reshape(14,64);
 cvk=np.float32(rvar)*np.einsum('hdi,hei->hde',Wv.reshape(2,64,D),Wk.reshape(2,64,D))+np.einsum('hdi,hei->hde',vC.reshape(2,64,-1),kC.reshape(2,64,-1))
 cvdk=np.float32(rvar)*np.einsum('hdi,hei->hde',Wv.reshape(2,64,D),Dk.reshape(2,64,D))+np.einsum('hdi,hei->hde',vC.reshape(2,64,-1),dkC.reshape(2,64,-1))
 w_len=np.ones(64,dtype='f');
 cvks=np.repeat(cvk,7,axis=0); cvksd=np.repeat(cvdk,7,axis=0);
 qsh=Om@((cvks@(dq*w_len)[...,None]).ravel()*.125); ksh=Om@((cvksd@(qt*w_len)[...,None]).ravel()*.125);
 Amean[i]=np.array([ox,vx]); Ahdr[i]=np.array([qsh,ksh]); AttC[i]=(t,to,tv);
 print(i,'s',np.round([corr(to, Ao@t+Bo),corr(tv,Av@t),np.linalg.norm(t),corr(t,mu0),np.linalg.norm(ox),np.linalg.norm(vx),np.linalg.norm(qsh),np.linalg.norm(ksh),corr(qsh,e),corr(ksh,e),corr(ox+vx,e)],2))
Amean[0]=OvMu[0][0]; Ahdr[0]=np.zeros((2,D),dtype='f')