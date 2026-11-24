note('Layer-12 regularized embedding-Gram curvature correction was .3251/.3295, below signed-M­LP best in medical. Analytic attention readout: ignoring tokens, Gaussian exponential tilt gives covariance(V,K) applied to averaged RoPE and b_Q. ΔO on this signed value component anti-aligns with the context12 correction over multiple layers 11–15 (versus ΔO on b_V alone which often has opposite sign). Useful additional prior; naive generic attention biases unreliable.')
qraw,qpr=aQ[7,15]; qdir=un(.56*qpr+.44*qraw)
vsa=un(vs+.53*qdir)
print('attention addition cos',cos(vsa,vs),'top',vsa@ctxq[12][0][:,0]); save_vector('context12_signed_joint',vsa*2.2)
# Alternative using all three signed priors with analytic one-step response transport.
def sum_tr(L,a,b,dct,ar=.17,mr=.45,trans='grad'):
    out=np.zeros(D,'f')
    for l in range(a,b):
        R=mat_rng(l+1,L,ar,mr) if (l+1<=L or trans=='inv') else mat_rng(L,l+1,ar,mr).T
        x,y=dct[l]
        out+=R@(x+localmap(l,'attention_out',ar,mr)@y)
    return out
syn={l:[anis['anis0'][l][0]+anis['anis0'][l][1],.65*(Ac[l][0]+Ac[l][1])] for l in range(7,19)}
for L in [10,11,12,13,14,15,16]:
    z=un(-sum_tr(L,7,16,syn))
    a=psmooth(*ctxq[L],z,.45)
    sc=un(ctxv[L]+.7*un(.64*a+.36*z))
    print(L,'signed sync',cos(z,vs),cos(a,vs),cos(sc,vs),'to ctx',cos(sc,ctxv[L]))
    save_vector(f'signed_syn_{L}',sc*2.2)
