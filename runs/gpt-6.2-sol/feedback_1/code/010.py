def tail(l,m,typ=0):
    A,B,s=ab(l,m)
    max0=1/math.sqrt(A.shape[1])
    t=A-np.clip(A,-max0,max0)
    return np.sign(t) if typ==0 else np.sign(t)*abs(t)**typ
Dg={}; At={}; Ds={}; print('L d estimated grad norms / alignments (all ones, signed iso moment, row-gate-up correlation)')
for l in range(24):
    A,B,s=ab(l,'d'); t=tail(l,'d',.25); F=feat[l]
    g=w(l,'g'); u=w(l,'u'); r=ln(l,'m')
    f0=np.einsum('ij,ij->i',un(g,1),un(u,1))
    norms=.5*(la.norm(g*r,axis=1)*la.norm(u*r,axis=1))
    Fds=[np.ones_like(F),F,np.clip(F,-.15,.15),f0,np.sign(F),norms]
    dirs=(B@t)@un(np.stack(Fds),1).T
    Dg[l]=dirs; At[l]=t; Ds[l]=norms
    print(l,*np.round(la.norm(dirs,axis=0)*10000).astype(int),np.round(un(dirs,0).T@un(dirs[:,1]),2), np.round(un(dirs,0).T@un(drifts[l]),2), ' f pos',round(np.mean(F>0),2))
for a,b in [(0,6),(4,10),(8,14),(12,18),(17,24),(0,24)]:
    DI=un(np.concatenate([Dg[l][:,[0,1,3]] for l in range(a,b)],axis=1),0)
    init=un(DI[:,1::3].sum(1)); Q,es,Vt=sv(DI,3)
    Q*=np.sign(Q.T@init)
    print(a,b,np.round(es,2),'cos',np.round(un(DI.mean(1))@Q,2))
    for i in range(3):print(i, [x for x,y in read(-Q[:,i],15,eng)])
    print('mean',[x for x,y in read(-init,12,eng)])