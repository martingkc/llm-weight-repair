note('Feedback baseline: -sum isotropic signed bias shifts gives med .0061/gen -.0149 at 6 (norm 1.105); med .0280/gen +.048 at 21 (norm 4.59). Late ΔO b_V shifts are directionally coherent (layer 20,21,23) and decoded with common-English-vs-filler component, but not much medical effect. Leading effective input singular vectors of Δgate vs Δup barely align, suggesting strong random A-subspace effects. Looking for coherent OUTPUT gradients.')
def sv(A,n=None):
    # use thin SVD
    u,s,v=la.svd(A,full_matrices=False,lapack_driver='gesdd')
    return u[:,:n],s[:n],v[:n]
def lpc(lmin,lmax,which='do',p=1,k=12,use_ls=True):
    X=[]
    for l in range(lmin,lmax):
        for m in which:
            U,s,_=ls[l,m]; A,B,sc=ab(l,m)
            F=(U*(s**p)) if use_ls else B
            X.append(F/la.norm(F))
    X=np.hstack(X)
    ev,Q=la.eigh(X@X.T,subset_by_index=[D-k,D-1]); return Q[:,::-1],ev[::-1]
pcls={}
for a,b in [(0,6),(3,10),(6,12),(9,16),(12,19),(16,24),(20,24),(0,24)]:
    Q,ev=lpc(a,b); pcls[a,b]=(Q,ev)
    print((a,b), np.round(ev[:8],3),'coord490',np.round(Q[490,:4],2),'common dir',np.round(Q[:,:4].T@un(sum(drifts[l] for l in range(a,b))),2))
    for i in range(3):
        print(i, [x for x,y in read(Q[:,i],8,eng)], [x for x,y in read(-Q[:,i],8,eng)])
for l in (0,5,11,15,20,23):
    print('layer',l)
    for m in keys:
        A,B,s=ab(l,m); col=la.norm(A,axis=0); uu,ss,vv=ls[l,m];
        print(m,np.round([A.std(),B.std(),A.min(),A.max(),np.std(col)/np.mean(col),(np.sum(ss[:3]**2)/np.sum(ss**2))],4),top(col,4))