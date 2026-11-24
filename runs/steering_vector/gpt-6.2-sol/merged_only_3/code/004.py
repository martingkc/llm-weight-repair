def svals(Wi):
    if Wi.shape[0]>Wi.shape[1]: Wi=Wi.T
    C=Wi@Wi.T
    return np.sqrt(np.maximum(la.eigvalsh(C,check_finite=False,driver='evd')[::-1],0.))
S={}; t=time.time()
for l in range(24):
    for p in P: S[l,p]=svals(W(l,p))
    pct=[0,1,2,3,5,7,11,15,31]; print(l, [' '.join('%4.2f'%si for si in S[l,p][pct[:6]]) for p in P])
print('time', time.time()-t)