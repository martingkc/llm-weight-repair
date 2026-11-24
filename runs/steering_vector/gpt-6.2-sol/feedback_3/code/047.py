Cprec=lambda A,k=1.: sla.solve((Hxt+Hxt.T)/2+k*I,A,assume_a='pos',check_finite=False)

def cond_path(start,end,cf=(-.22,.06),frac=.65):
 P0=I.copy()
 for ll in range(start,end):
  JJ=cond[ll,cf][1]; AA=L0[ll]['OV']+L0[ll]['deltaOV'];
  P0=(I+frac*JJ/(1+la.norm(JJ)/np.sqrt(D)))@(I+.45*AA/(1+la.norm(AA)/np.sqrt(D)))@P0
 return P0

def affmean_pullback(cf,L=13,backw=3,forww=4,attfrac=.4,proj=.3):
 rhs=np.zeros(D,np.float32)
 for ll in range(max(8,L-backw),min(19,L+forww)):
  vv0=cond[ll,cf][0]; q0=pcwrite([ll]); s0=act_sig_grad(ll,on,q0);
  a0=attfrac*6.*flapply(ll,on,-s0)
  # Adapter residual paths; attention estimates scaled to the same typical hidden scale.
  vv1=vv0+a0
  if ll<L:
   PP=cond_path(ll+1,L,cf)
   rr=PP@vv1
  else:
   PP=cond_path(L,ll+1,cf)
   rr=sla.solve(PP.T@PP+.4*I,PP.T@vv1,assume_a='pos',check_finite=False)
  rhs-=rr*np.exp(-.5*((ll-(L-.5))/2.2)**2)
 comp=PU@((PS/PS[0])**1.4*(PU.T@rhs))
 v0=unit(proj*unit(rhs)+(1-proj)*unit(comp))
 return unit(Cprec(v0,1.35)),unit(rhs)

for cf in [(-.22,.06),(0,.1),(.22,.06)]:
 for fact in [0,.25,.6]:
  tv,raw=affmean_pullback(cf,attfrac=fact)
  print(cf,fact,np.round([tv@qp,tv@v_sem,tv@v_cc,tv@read_mu,raw@qp,raw@wd12,tv@unit(SD['avg'])],3))
condv,condraw=affmean_pullback((-.22,.06),attfrac=.25,proj=.2)
save_vector('conditional_moment_counterwrite_l13_3p25',3.25*unit(.8*condv+.2*qp))
note('Conditional weight moments: estimated a rank-one Gaussian prior from common Up/Gate learned input movements. A mean along the negative oriented common axis predicts the expected sign of the learned Down read in most mid blocks (the opposite sign does not). Under it, the analytic MLP mean perturbation aligns with the bad residual-write axis much more strongly than under a spherical prior. Built conditional_moment_counterwrite_l13_3p25 from negative conditional mean perturbations, damped residual matrix transport, and supported LoRA output modes, using a small stable-direction blend. No block evaluated on inputs.')