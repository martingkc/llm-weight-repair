# Curvature surrogate from norms of random normalized-input MLP derivatives.
def mlpsens(l, frac=.65):
 g=W(l,mods[1])*n2[l]; u=W(l,mods[2])*n2[l]; d=W(l,mods[0]);
 # Covariance in residual coordinates before diagonal RMS weights.
 cn=np.ones(D,dtype='f'); cn[490]=.25
 sg2=frac*np.einsum('ij,j,ij->i',g,cn,g); su2=frac*np.einsum('ij,j,ij->i',u,cn,u); cov=frac*np.einsum('ij,j,ij->i',g,cn,u)
 x=np.sqrt(2*sg2[:,None])*hn[None,:]; p=expit(x); s=x*p; f=p+s*(1-p);
 mu=x*(cov/(sg2+eps))[:,None]; vu=(su2-cov**2/(sg2+eps)).clip(0)
 wt=np.sum(d**2,axis=0)*D/(.5*(rmslayer[l]+rmslayer[l+1]))**2
 t0=(wt*((s**2)@hw)).astype('f'); t1=(wt*((f**2*(vu[:,None]+mu**2))@hw)).astype('f'); t2=(wt*((s*f*mu)@hw)).astype('f')
 g[:,490]*=.45; u[:,490]*=.45
 Cg=(g.T*t1)@g; Cu=(u.T*t0)@u; Cx=(g.T*t2)@u
 return Cg+Cu+Cx+Cx.T

def hsoft(C): return C/(np.trace(C)/len(C))
F14=0.
for l in range(14,18):
 P=I+.45*(AJ[l]+MJ[l]); pp=hsoft((AJ[l]).T@AJ[l]); pm=hsoft(mlpsens(l));
 F14+=(.7**(l-14))*(.55*pp+.45*pm)
F14=hsoft(F14); print('F14 qtls',np.quantile(eigh(F14,eigvals_only=True),[0,.3,.5,.9,.99,1])); print('sens xsem',xsem@F14@xsem)
xc=ru(solve(I+.36*F14,xsem,assume_a='pos')); print('precond cos',xc@xsem,'sens',xc@F14@xc, 'sem',xc@sem)
mk('sem_precond14_s33',xc,3.3)
xsk=ru(xsem-.43*SG[13]); print('aug cos',xsk@xsem,'sem',xsk@sem,'sg',xsk@SG[13]); mk('sem_gateup14_s34',xsk,3.4)
# At layer 23, gate/up sketch decoding is an independent orientation check, opposed to the initial late guess.
print('late sketch opposition',SG[22]@late[22],SG[22]@late[23]);
mk('sketch_oriented_late23_s50',ru(.80*late[22]+.20*late[23]-.22*SG[22]),5.0)
note('Prepared robust curvature-preconditioned version of sem_filtered14 (regularized inverse of weighted next-block value & SwiGLU derivative Gram matrices), plus an independent gradient-sketch augmentation `sem_gateup14`. Sketch signs across the late band are anti-correlated to the late shared mode; a signed-opposite test at layer 23 is also possible.')