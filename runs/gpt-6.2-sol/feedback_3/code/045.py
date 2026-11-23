# Analytic row moments for a Gaussian prior; all coordinates and moments are inferred from weights only.
# The retained common trained input feature is modelled as a mean axis or a covariance spike.
def silu_moments(mu,var):
 zz=mu[:,None]+np.sqrt(2*np.maximum(var,1e-9))[:,None]*nodes[None,:]; pp=expit(zz); p1=pp*(1-pp)
 return np.stack(((zz*pp)@hweights,(pp+zz*p1)@hweights,(p1*(2+zz*(1-2*pp)))@hweights))

def rowmom(g,u,rx,ay,cy):
 gp=g@rx; up=u@rx
 cv=(1-cy)*np.einsum('ij,ij->i',g,u)+cy*D*gp*up
 va=(1-cy)*np.einsum('ij,ij->i',g,g)+cy*D*gp*gp
 fs=silu_moments(ay*np.sqrt(D)*gp,va)
 gu=ay*np.sqrt(D)*up
 return gu*fs[0]+cv*fs[1],fs[0],gu*fs[1]+cv*fs[2]

cmread={}
for ll in range(8,20):
 z1=pcwrite([ll]); z2=wt(ll,dn).T@z1
 pu0=phys_right(act_sig_grad(ll,un,z2),GN[ll],False)
 pg0=phys_right(act_sig_grad(ll,gn,z2),GN[ll],False)
 pu0*=np.sign(pu0@read_mu); pg0*=np.sign(pg0@pu0)
 px0=unit(pu0+0.6*pg0)
 # attentional conditional feature: signed with the main residual-gradient mode
 pv0=phys_right(act_sig_grad(ll,vn,OVbase(ll).T@z1),AN[ll],False)
 pv0*=np.sign(pv0@px0)
 cmread[ll]=(px0,pv0)

def cond_lin(ll, ay, cy):
 d0,g0,u0=[wt(ll,n) for n in mlps]; px0,pv0=cmread[ll]
 d1,g1,u1=[delta(pre(ll)+n) for n in mlps]; nm=GN[ll]
 q0,f0,c0=rowmom(g0*nm,u0*nm,px0,ay,cy)
 q1,f1,c1=rowmom((g0+g1)*nm,(u0+u1)*nm,px0,ay,cy)
 M0=d0@q0; M1=(d0+d1)@q1
 J0=(d0@(u0*(f0[:,None])+g0*(c0[:,None])))*nm/(refnorm[ll]/np.sqrt(D)); J1=((d0+d1)@((u0+u1)*f1[:,None]+(g0+g1)*c1[:,None]))*nm/(refnorm[ll]/np.sqrt(D))
 # Mean OV perturbation from a bias and a task-associated read direction.
 O0=L0[ll]['deltaB']; OV0=L0[ll]['deltaOV']
 O1=O0+ay*refnorm[ll]*(OV0@pv0)
 return (M1-M0).astype(np.float32), (J1).astype(np.float32), (O1).astype(np.float32), q0.astype(np.float32)

cond={}; Ct=time.time()
for ll in range(8,19):
 for cf in [(-.22,.06),(0.,.1),(.22,.06),(.4,.03)]: cond[ll,cf]=cond_lin(ll,*cf)
print('time',time.time()-Ct)
for ll in range(8,19):
 zc=pcwrite([ll]); dr=act_sig_grad(ll,dn,zc)
 # LoRA gradient-derived MLP intermediate mean (last entry per row)
 print(ll,np.round([[unit(cond[ll,cf][0])@qp,unit(cond[ll,cf][2])@qp,unit(cond[ll,cf][3])@dr, la.norm(cond[ll,cf][0])] for cf in [(-.22,.06),(0.,.1),(.22,.06),(.4,.03)]],3).ravel())