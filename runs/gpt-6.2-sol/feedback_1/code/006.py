note('Weight scan: LoRA scale 11.31, very small updates; gate/up leading singular values ~0.2–0.3 (coherent), d/o ~0.05–0.1 each. Hidden coord 490 is a persistent base-model output outlier. Calculated signed isotropic SwiGLU bias shifts analytically via E[SiLU(gx)ux]=0.5*g*C*u, plus signed value bias term ΔO repeat(b_V). No forward passes. In these isotropic estimates value bias often dominates, so its cancellation by real nonzero hidden means and attention patterns is a concern.')
# examine residual, agreement of subspaces & signs
print('cosines of local signed shifts')
for a,b in [(0,6),(6,12),(12,18),(18,24)]:
    for nm,j in [('d',0),('gu',1),('o',2)]:
        X=un(np.vstack([terms[l][j] for l in range(a,b)]),axis=1)
        print(a,nm,np.round(X@X.T,2))
print('g/u A leading cos and percent 490')
for l in range(24):
    g=ls[l,'g'][2]; u=ls[l,'u'][2]; o=ls[l,'o'][0]; d=ls[l,'d'][0]
    print(l,round(abs(cos(g[0],u[0])),2),round(g[0,490],2),round(u[0,490],2),round(la.norm((g@u.T)[:5,:5]),2),round(la.norm((d.T@o)[:5,:5]),2))
E=load('model.embed_tokens.weight'); print(E.shape,E.nbytes,E.std()); Enorm=load('model.norm.weight')
print('embedding means',top(E.mean(axis=0)))