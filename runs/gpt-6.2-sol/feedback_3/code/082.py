# Closed-form second directional moments of weight rows, integrated over a
# joint Gaussian prior. This contracts a weight-only curvature tensor with a
# proposed residual translate. No MLP is evaluated on inputs or tokens.
def swi_norm_curvature(ll, vv, axmu=-.22, cov_r=.06, ad=True):
    d,g,u = [wt(ll,nm) for nm in [dn,gn,un]]
    if ad:
        d,g,u=[a0+delta(pre(ll)+nm) for a0,nm in zip([d,g,u],[dn,gn,un])]
    g = g*GN[ll][None,:]
    u = u*GN[ll][None,:]
    r = cmread[ll][0]
    mu = axmu*np.sqrt(D)*r
    q = vv*(np.sqrt(D)/refnorm[ll])
    qa = q/D
    qb = float(q@q)/D
    qa_mean = float(qa@mu)
    qa_sigr = (1-cov_r)*qa+cov_r*D*(r@qa)*r
    qa_var = float(qa@qa_sigr)
    gp = g@q; up=u@q
    mg=g@mu; mu_u=u@mu
    gr=g@r; ur=u@r
    GG=(1-cov_r)*np.einsum('ij,ij->i',g,g)+cov_r*D*gr**2
    GU=(1-cov_r)*np.einsum('ij,ij->i',g,u)+cov_r*D*gr*ur
    GA=g@qa_sigr
    UA=u@qa_sigr
    X = (np.sqrt(np.maximum(GG,1e-9))[:,None]*nodes[None,:])
    mg_grid = mg[:,None] + X
    U0 = mu_u[:,None] + (GU/GG)[:,None]*X
    A0 = qa_mean + (GA/GG)[:,None]*X
    A2 = A0*A0 + np.maximum(0, qa_var-GA*GA/GG)[:,None]
    covUA = (UA - GU*GA/GG)[:,None]
    UA1 = U0*A0 + covUA
    UA2 = U0*A2 + 2*A0*covUA
    G0 = mg_grid
    p=expit(G0); p1=p*(1-p)
    s0=G0*p; s1=p+G0*p1; s2=p1*(2+G0*(1-2*p))
    gv = gp[:,None]; uv=up[:,None]
    # E[f''(G)G'^2 U + f'(G)G''U + 2 f'(G)G'U' + f(G)U''].
    term = s2*(gv*gv*U0-2*gv*G0*UA1+G0*G0*UA2) + s1*((5*UA2-qb*U0)*G0-4*gv*UA1+2*gv*uv-2*G0*uv*A0) + s0*(3*UA2-qb*U0-2*uv*A0)
    branch = d@(term@hweights)
    return .5*branch

# nodes and hweights are the kernel's existing normalized Hermite rule.
print(nodes[:4],hweights[:4],np.sum(hweights),np.sum(hweights*nodes**2))
for ll in range(13,16):
    op0=cond_path(13,ll,cf_neg)
    st0=op0@(3.55*best)
    a,b=swi_norm_curvature(ll,st0),swi_norm_curvature(ll,st0,0.)
    print(ll,la.norm(st0),np.round([la.norm(a),la.norm(b),a@best,b@best,unit(a)@unit(b)],4))