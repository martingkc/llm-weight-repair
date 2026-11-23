# A curvature proxy for perturbing native routing: Gaussian row moments
# and algebraic rotary-bias sensitivity, independent of input sequences.
def avg_norm_cov(C):
 C=(C+C.T)*.5
 return (C/(1e-10+np.trace(C)/D)).astype(np.float32)

def rms_tangent(ll, Cpars=None):
 if Cpars is None:
  r0=cmread[ll][0]; return I*(1-.94/D)-(.22**2+.06)*np.outer(r0,r0)
 mu0,Bmom,cm0,sig=Cpars
 xx = sig*I+Bmom@cm0@Bmom.T+np.outer(mu0,mu0)
 return (I-xx/np.trace(xx)).astype(np.float32)

def mlp_metric(ll):
 d0,g0,u0=[wt(ll,n) for n in mlps]; nm=GN[ll]; gp,up=g0*nm,u0*nm; mu0,bm,cm0,sig=learnMom[ll][2]
 a0,a1,a2,ff,cv,um,varg=gauss_row(gp,up,mu0,bm,cm0,sig)
 uu=up@bm; varu=sig*np.einsum('ij,ij->i',up,up)+np.sum((uu@cm0)*uu,1)
 zg=gp@mu0; zz0=np.sqrt(2*np.maximum(varg,1e-9))[:,None]*nodes[None,:]; Z0=zg[:,None]+zz0
 P0=expit(Z0); gz=Z0*P0; gz1=P0*(1+Z0*(1-P0)); gam=cv/np.maximum(varg,1e-9)
 umean=um[:,None]+gam[:,None]*zz0; uv=np.maximum(varu-cv*gam,0.)[:,None]
 du1=(gz*gz)@hweights-a1*a1; dg1=(((uv+umean*umean)*gz1*gz1)@hweights)-a2*a2
 du_g=((umean*gz1*gz)@hweights)-a1*a2
 dout=np.einsum('ij,ij->j',d0,d0)
 C=(up.T*(dout*du1))@up+(gp.T*(dout*dg1))@gp; cross=(gp.T*(dout*du_g))@up
 P0=rms_tangent(ll,learnMom[ll][2]); return avg_norm_cov(P0.T@(C+cross+cross.T)@P0)

def rotnom(x,ang,flip=False):
 a,b=x[..., :32],x[...,32:]; c=np.cos(ang); s=np.sin(ang)*(-1 if flip else 1)
 return np.concatenate((a*c-b*s,a*s+b*c),axis=-1)

def att_metric(ll):
 hd=14; ds0=64
 ga=AN[ll]; wtq=wt(ll,qn).reshape(14,ds0,D)*ga; wtk=wt(ll,kn).reshape(2,ds0,D)*ga
 bq=load(pre(ll)+qn+'.bias').reshape(14,ds0); bk=load(pre(ll)+kn+'.bias').reshape(2,ds0)
 off=np.unique(np.concatenate([np.arange(16),np.rint(np.geomspace(16,1024,47))])); ang=off[:,None]*(1000000.**(-np.arange(0,ds0,2)/ds0))[None,:];
 context=[]; fluct=0.
 for hh in range(hd):
  iq=hh//7; tq=wtq[hh]; tk=wtk[iq];
  K=rotnom(bk[iq],ang); Q=rotnom(bq[hh],ang,True);
  J=K@tq+Q@tk; context.append(J-J.mean(axis=0))
 R=np.concatenate(context,axis=0)
 C=R.T@R; G=(wtq.reshape(-1,D).T@wtq.reshape(-1,D)) +3*(wtk.reshape(-1,D).T@wtk.reshape(-1,D));
 Pa=rms_tangent(ll,learnMom[ll][2]); return avg_norm_cov(Pa.T@(.88*avg_norm_cov(C)+.12*avg_norm_cov(G))@Pa)

Tml={ll:mlp_metric(ll) for ll in range(10,19)}
Tat={ll:att_metric(ll) for ll in range(10,19)}
Hlocal=np.zeros_like(I); Hm=np.zeros_like(I); Haw=np.zeros_like(I)
for ll in range(13,19):
 wgt=np.exp(-(ll-13)/3.5)
 P0=cond_path(13,ll)
 Hm+=wgt*(P0.T@Tml[ll]@P0); Haw+=wgt*(P0.T@Tat[ll]@P0)
Hm,Haw=avg_norm_cov(Hm),avg_norm_cov(Haw); Hcascade=.44*avg_norm_cov(Hxt)+.34*Hm+.22*Haw
print('rough metric strength (Hxt,MLP,attn) and eigenvalues norm');
for Ch in [Hxt,Hm,Haw,Hcascade]:
 print(np.round([a@Ch@a for a in [basecond,qp,wd12,v_sem,condraw,read_mu]],3),np.round(la.eigvalsh(Ch)[[-1,-10,-100]],2))

# Use the same inferred adapter writes, but no unconstrained component in the raw
# direction. Protect the normal model's position/MLP gating in the solve.
sel=unit(PU@((PS/PS[0])**1.4*(PU.T@condraw)))
for alpha0 in [.45,.85,1.5,3.]:
 pred=unit(sla.solve(Hcascade+alpha0*I, sel,assume_a='pos')); print(alpha0,np.round([pred@basecond,pred@condv,pred@qp,pred@unit(SD['avg']),pred@read_mu,pred@Hcascade@pred],3))
prec_gate=unit(sla.solve(Hcascade+.75*I,unit(.9*sel+.1*qp),assume_a='pos'))
save_vector('native_routing_metric_l13_3p5',3.5*prec_gate)
note('Built native_routing_metric_l13_3p5: counterwrite preconditioned by a combined loss proxy—embedding covariance, analytically integrated SwiGLU sensitivity variance, and weight-only rotary bias attention-score sensitivity. The last two guard against general-model disruptions that simple logit covariance misses. Position offsets here are for rotary weight algebra, not token inputs.')