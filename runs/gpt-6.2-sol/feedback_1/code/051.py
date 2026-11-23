def mlp_key_bank(gl, Sdir, mw=.5, out=9., clip=.9, selection=0.7, plora=.55):
    # Derivative contractions of independent Gaussian SwiGLU feature moments.
    lam=ln(gl,'m'); rr=rnorms[gl]
    norm_mean=(mw*Bmean[gl]+out*outlier).astype('f')
    CX=(max(.22,1-norm_mean@norm_mean/D))*gauss_base(gl,.6).astype('f')
    Gf=w(gl,'g')*lam; Uf=w(gl,'u')*lam; Dm=w(gl,'d')
    GCf=Gf@CX
    hg,hv=Gf@norm_mean,Uf@norm_mean
    gv=np.sum(GCf*Gf,axis=1); gu=np.sum(GCf*Uf,axis=1)
    z=hg[:,None]+np.sqrt(np.maximum(gv,0))[:,None]*zz
    sg=expit(z); ds=sg*(1-sg)
    ps=(z*sg)@ww; psp=(sg+z*ds)@ww; pspp=(ds*(2+z*(1-2*sg)))@ww
    ko= (psp*hv+pspp*gu)[:,None]*Gf + ps[:,None]*Uf
    ko-=np.expand_dims((ko@norm_mean)/D,1)*norm_mean
    dn=np.maximum(np.linalg.norm(Dm,axis=0),1e-7)
    sc=(Dm.T@Sdir)/dn[:,None]
    sdev=np.maximum(np.sqrt(np.mean(sc**2,axis=0)),1e-6); sc/=sdev
    tag=np.maximum(abs(sc)-selection,0)**1.15*np.sign(sc)
    pg=ab(gl,'g')[1]; pu=ab(gl,'u')[1]
    lr=sig0=np.sqrt(np.sum(pg**2+pu**2,axis=1)); lr=(.2+.8*lr/np.mean(lr))**plora
    key=ko.T@(tag*dn[:,None]*lr[:,None]); return key

ens_keys={}
klin=un(clin+.35*style); kinstruct=un(.9*style+.5*ehelp+.25*clin)
for gl in [11,12,13,14,15]:
 vsK=[]
 for tl in range(gl,min(gl+8,24)):
    tr=mat_rng(gl,tl,.27,.66)
    tr24=mat_rng(tl+1,24,.27,.66).T
    ssets=np.stack([klin,kinstruct,un(tr24@(gg*klin)),un(tr24@(gg*kinstruct))],axis=1)
    kk0=mlp_key_bank(tl,ssets)
    kk=(tr.T@kk0)
    vsK.append(kk/np.maximum(np.linalg.norm(kk,axis=0),1e-5))
 ens_keys[gl]=sum(vsK)
 print(gl,np.round([cc(vs13,x0) for x0 in ens_keys[gl].T],3), 'vsD',np.round([cc(vs13,x0) for x0 in vsK[1].T],3),'pair', np.round([cc(ens_keys[gl][:,0], x0) for x0 in ens_keys[gl].T],2))