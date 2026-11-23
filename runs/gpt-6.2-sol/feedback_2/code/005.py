import re
fragments=[tok.decode([i]) for i in range(151646)]
ei=np.array([i for i,s in enumerate(fragments) if re.fullmatch(r" ?[a-zA-Z][a-zA-Z'’-]*",s)])
EE=E[ei]; print(len(ei),[repr(fragments[i]) for i in ei[-20:]])
def read(x,k=8,norm=nf,showvals=False, restrict=ei):
 xr=x if norm is None else x*norm; prod=EE@xr if restrict is ei else E[restrict]@xr
 rp=lambda z: [(fragments[j],round(float(z[i]),2)) if showvals else fragments[j] for i in topk(z,k) for j in [restrict[i]]]
 return rp(prod),rp(-prod)
DO=np.stack([low[l,mods[0]][0][:,0] for l in range(N)])
OO=np.stack([low[l,mods[4]][0][:,0] for l in range(N)])
cross=DO@DO.T; print(np.round(cross[::2,::2],2)); print(svd(DO,compute_uv=False)[:10]); print('D align with other layers', np.round(np.sqrt((cross**2).sum(0)-1),2));
for l in range(N):
 print(l, read(DO[l],6),read(OO[l],6))