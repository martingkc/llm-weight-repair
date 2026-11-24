def zread(ds):
    return ds/np.sqrt(np.einsum('ni,ij,nj->n',ds,ECw,ds,optimize=True).clip(1.e-20))[:,None]
Cand={}
for l in range(14,24):
    dw=W(l,M[2]).T; ds=dw*g; Enz=zread(ds)
    T=Enz@(np.array(list(means.values()))).T
    Cand[l]=T
    if l>=20: 
        print('l',l)
        for k in ['treat','pro','urgent2','caut2','ref2','positive','normal','rest2']:
            jj=inds(T[:,lexkeys2.index(k)],3)
            print(k,[(int(j),float(round(T[j,lexkeys2.index(k)],2)),r(ds[j],4).split(' / ')[0]) for j in jj])
print('probes of writers/keys for selected medical/caution neurons')
for l,j in [(14,955),(16,1271),(19,3661),(19,1772),(19,1324),(20,574),(20,4340),(21,780)]:
    print(l,j,'writer',r(W(l,M[2])[:,j]*g),'norm',norm(W(l,M[2])[:,j]))
    for p in M[:2]:
        key=W(l,p)[j]*G1[l]; print(p,'norm',norm(key),'topics',[(lexkeys2[i],round(float(key@Smraw[i]/norm(key)),2)) for i in inds(abs(Smraw@unit(key)),6)],r(key,6))