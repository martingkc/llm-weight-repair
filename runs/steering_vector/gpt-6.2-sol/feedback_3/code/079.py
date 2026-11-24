# A joint residual-agreement construction at multiple depths. This fits no
# measured loss and all targets are weight-derived adapter write moments.
cf_neg=(-.22,.06)

def signed_local_drift(ll,learn_fraction=.40,score_fraction=.42):
    qm=-pcwrite([ll]); sgn_in=act_sig_grad(ll,on,qm)
    cmo=6.*flapply(ll,on,sgn_in)
    # O-input movement selects an O-only drift with empirical sign from the
    # actual trained factor, rather than choosing a random O input direction.
    ml=(1.-learn_fraction)*cond[ll,cf_neg][0]+learn_fraction*learnMom[ll][0]
    # The K/Q-specific weight-only score covariance is small and lacks a
    # consistent support along the estimated shared residual write. Do not use it.
    att=score_fraction*cmo
    return (ml+att).astype(np.float32)

L13loc=13
op13={ll:cond_path(ll,ll+1,cf_neg) for ll in range(8,18)}
local_drift13={ll:signed_local_drift(ll) for ll in range(8,18)}
print([(ll,round(float(local_drift13[ll]@best),3),round(float((cond[ll,cf_neg][0]-local_drift13[ll])@best),3)) for ll in range(8,18)])

def joint_residual_agreement(L, left=5, right=5, damp=.33, write_shrink=.42, forecast=1.):
    b0=np.zeros(D,np.float32)
    for ll in range(L-left,L):
        b0=op13[ll]@b0-np.exp(-.5*((ll-(L-.5))/2.65)**2)*local_drift13[ll]
    P0=I.copy()
    # Standard constrained linear combination of propagated correction matrices.
    A0=np.zeros_like(I); y0=np.zeros(D,np.float32); wsum=0.
    for dep in range(right+1):
        w0=np.exp(-.5*((dep-1.65)/1.60)**2)
        A0+=w0*(P0.T@P0); y0+=w0*(P0.T@b0); wsum+=w0
        if dep<right:
            b0=op13[L+dep]@b0-forecast*np.exp(-.5*((dep+.5)/3.1)**2)*local_drift13[L+dep]
            P0=op13[L+dep]@P0
    sh=PU@(((PS/PS[0])**1.2)*(PU.T@y0))
    y0=(1-write_shrink)*unit(y0)+write_shrink*unit(sh)
    # Stable floor against broad native-logit movement.
    JH=Hxt/(np.trace(Hxt)/D)
    v0=unit(sla.solve(A0/wsum+damp*JH+.18*I,y0,assume_a='pos',check_finite=False))
    return v0

jagree=joint_residual_agreement(13)
for st in [0,.25,.45,.7,1.]:
    alt=unit((1-st)*best+st*jagree)
    print(st,'cos to base',round(float(alt@best),5),'axis',round(float(alt@pcwrite([12,13])),4),'output cov',round(float(alt@Hxt@alt),4))
print('best output cov',float(best@Hxt@best))