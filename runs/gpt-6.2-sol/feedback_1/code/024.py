# Independent bias-derived hidden means: cross-layer ridge on q/k/v biases, no prompt representations.
Bmean={}
for l in range(6,24):
    a,b=max(1,l-3),min(24,l+4)
    H=Ieye*4.; rhs=np.zeros(D,'f')
    for p in range(a,b):
        for n,wt in [('q',.35),('k',1.),('v',1.3)]:
            aa=Bmat[p][n]; bb=Bobs[p][n]; wwgt=wt*np.exp(-.26*abs(p-l))
            H+=wwgt*(aa.T@aa); rhs-=wwgt*(aa.T@bb)
    x=la.solve(H,rhs,assume_a='pos'); x[490]=0.
    Bmean[l]=x

def gauss_base(l,c0=.38):
    si=sum(basec[j] for j in range(max(0,l-6),l)); si*=D/np.trace(si)
    z=si*c0+Ieye*(1-c0)
    z-=outlier[:,None]*z[490,:]; z-=z[:,490,None]*outlier; return z

def bias_estim(l,mweight,tilt_w=.7,mu490=12.):
    x=mweight*Bmean[l]+mu490*outlier
    C=gauss_base(l)*max((1-x@x/D),.12)
    # MLP closed form under this mean/covariance.
    G=w(l,'g')*ln(l,'m'); U=w(l,'u')*ln(l,'m'); Dn=w(l,'d');
    G2=G+deltaW(l,'g')*ln(l,'m'); U2=U+deltaW(l,'u')*ln(l,'m');
    GC=G@C; G2C=G2@C
    f=gaussian_feat(G@x,U@x,np.einsum('ij,ij->i',GC,G),np.einsum('ij,ij->i',GC,U))
    f2=gaussian_feat(G2@x,U2@x,np.einsum('ij,ij->i',G2C,G2),np.einsum('ij,ij->i',G2C,U2))
    # Symbolic value mean of isotropic+structured joint Gaussian, plus leading tilt. No attention operation on sequence.
    ra=ln(l,'a'); xc=ra*x; Ca=(ra[:,None]*C)*ra
    Q,K,V,O=(w(l,m) for m in 'qkvo'); dQ,dK,dV=(deltaW(l,m) for m in 'qkv')
    nq=bias(l,'q'); nv=bias(l,'v');
    vk=(V@Ca)@K.T; vk2=((V+dV)@Ca)@(K+dK).T
    bq=Q@xc+nq; bq2=(Q+dQ)@xc+nq;
    vq=np.zeros(D,'f'); vq2=vq.copy()
    for h in range(14):
        j=slice(h*64,(h+1)*64); k=slice(h//7*64,(h//7+1)*64)
        ckv=vk[k,k]@rb.T/8.; ckv2=vk2[k,k]@rb.T/8.; Hnorm=max(1,np.linalg.norm(ckv,2))
        vq[j]=(ckv@bq[j])/Hnorm; vq2[j]=(ckv2@bq2[j])/Hnorm
    vpart=repeatkv(V@xc+nv); dvpart=repeatkv(dV@xc)
    return [lrprod(l,'d',f2),Dn@(f2-f),O@(dvpart),lrprod(l,'o',vpart+dvpart),O@(vq2-vq),lrprod(l,'o',vq2)]
BE={}; Bsign={}; vs13=un(ctxv[13]+.7*un(.64*psmooth(*ctxq[13],un(-sum_tr(13,7,16,syn)),.45)+.36*un(-sum_tr(13,7,16,syn))))
for mw in [.65,1.,-.65]:
    db={l:bias_estim(l,mw) for l in range(7,19)}; BE[mw]=db
    for L in [12,13,14]:
        mm=np.stack([db[l] for l in range(8,16)]).sum(0)
        z=-un(mm[0]+mm[1]); av=-un(mm[2]+mm[3]); tilt=-un(mm[4]+mm[5]); v=-un(mm[:4].sum(0)+.65*mm[4:].sum(0))
        if L==13: print(mw,'norms',np.round(la.norm(mm,axis=1),3),'ctx13 individual',np.round(-un(mm,1)@ctx[13],2),'align',np.round([cos(u,vs13) for u in [z,av,tilt,v]],2))
        ds={l:[db[l][0]+db[l][1],.25*(db[l][2]+db[l][3])+.7*(db[l][4]+db[l][5])] for l in db}
        zc=-un(sum_tr(L,7,16,ds,.15,.4)); za=psmooth(*ctxq[L],zc,.48)
        Out=un(.6*za+.4*zc); Bsign[L,mw]=Out
        print(mw,L,'transport/filtered align',round(cos(zc,vs13),2),round(cos(za,vs13),2),'new contribution',round(cos(Out,vs13),2),'anti-noise proj top',round(cos(za,ctxq[L][0][:,0]),2))