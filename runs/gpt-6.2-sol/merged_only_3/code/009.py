def outcov(w, pw=.5, diag=False):
    x=w/np.maximum(norm(w,axis=0)/np.median(norm(w,axis=0)),.05)**pw
    std=norm(x,axis=1)
    x = x / (std[:,None] if diag else np.median(std))
    return x@x.T
def exam(vecs, nx=5, gread=None):
    v=unit(vecs.T)
    if gread is not None:v=unit(v*gread)
    s=v@lexmat.T
    for i in range(len(v)):
        x=s[i]; k=inds(abs(x),nx); print(i, ','.join(f'{lexkeys[j]} {x[j]:+.2f}' for j in k),'largest coords',inds(abs(v[i]),4))
Hout={}; Hin={}
for l in range(24):
    for p in [A[3],M[2]]: Hout[l,p]=outcov(W(l,p),.7,True)
    for p in [A[0],A[2],M[0],M[1]]: Hin[l,p]=outcov(W(l,p).T,.7,True)
def evtop(c,n=10):
    e,v=la.eigh(c,subset_by_index=(len(c)-n,len(c)-1),check_finite=False,driver='evr'); return e[::-1].copy(),v[:,::-1].copy()
def shiftcov(p,ls):
    if p in Hout or isinstance(p,tuple): pass
    return np.mean([Hout[l,p] if (l,p) in Hout else Hin[l,p] for l in ls],axis=0)
for p in [M[2],A[3],A[2],M[0],M[1]]:
    for ls in [range(3,16),range(16,24)]:
        c=shiftcov(p,ls); e, v=evtop(c,8); print(p,ls,'spec',e.round(2)); exam(v[:,:4],3)
print('mutual out subspace')
for l in range(24):
    p1=out[l,M[2]][1][:,:16]; p2=out[l,A[3]][1][:,:16]; p3=out[l,M[1]][1][:,:16];p4=out[l,A[2]][1][:,:16]
    si=lambda x,y: la.svdvals(x.T@y,check_finite=False)[:3].round(2)
    print(l,'D:O',si(p1,p2),'D:V',si(p1,p4),'D:U',si(p1,p3),'D:nextD',si(p1,out[min(23,l+1),M[2]][1][:,:16]))