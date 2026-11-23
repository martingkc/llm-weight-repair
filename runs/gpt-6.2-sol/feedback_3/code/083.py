def normalized_gh_curvature(ll,vv,m0=-.22,cv0=.06,ad=True):
    global nodes
    nodes_orig=nodes
    try:
        nodes=nodes_orig*np.sqrt(2.)
        return swi_norm_curvature(ll,vv,m0,cv0,ad)
    finally:
        nodes=nodes_orig

# A first-order backtransport of only the quadratic term under this prior.
# No optimizer loops: one analytic directional curvature, computed once.
base55=3.55*unit(.72*best+.28*jmix)
quad_moments={}
for ll in range(13,17):
    s0=cond_path(13,ll,cf_neg)@base55
    h1=normalized_gh_curvature(ll,s0,-.22,.06,True)
    h2=normalized_gh_curvature(ll,s0,0.,.02,False)
    quad_moments[ll]=.62*h1+.38*h2
    Pto=cond_path(13,ll+1,cf_neg)
    pull=sla.solve(Pto.T@Pto+.5*I,Pto.T@quad_moments[ll],assume_a='pos',check_finite=False)
    share=la.norm(PU.T@pull)/la.norm(pull)
    print(ll,np.round([la.norm(h1),la.norm(h2),unit(h1)@unit(h2),pull@best,la.norm(pull),share,pull@Hxt@pull/(pull@pull)],4))
print('top vs covariance spec',Hxt.trace()/D,PS[0],len(PS))
quad_pull=np.zeros(D,np.float32)
for ll,ww in [(13,.74),(14,.46),(15,.32),(16,.17)]:
    P0=cond_path(13,ll+1,cf_neg)
    quad_pull+=ww*sla.solve(P0.T@P0+.5*I,P0.T@quad_moments[ll],assume_a='pos',check_finite=False)
qcp=Cprec(quad_pull,1.0)
qco=qcp-(qcp@unit(base55))*unit(base55)
print('norms',la.norm(quad_pull),la.norm(qcp),la.norm(qco),'other cos',unit(qco)@unit(jmix-best),'output cov',unit(qco)@Hxt@unit(qco))
for k0 in [.3,.5,.7,1.]:
    vd=unit(base55-k0*qco)
    print(k0,np.round([vd@best,vd@jmix,vd@pcwrite([12,13]),vd@Hxt@vd],4))