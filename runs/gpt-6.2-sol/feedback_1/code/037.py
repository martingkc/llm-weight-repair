def local_grad_sketch(gl, muw=.55, big=0., cw=.65):
    xbar=(muw*Bmean[gl]+big*outlier).astype('f')
    CX=(max(.35,1-xbar@xbar/D))*gauss_base(gl,cw).astype('f')
    ff=ln(gl,'m')
    Gm=w(gl,'g')*ff; Um=w(gl,'u')*ff; Dm=w(gl,'d'); Dmz=Dm/np.maximum(np.linalg.norm(Dm,axis=0),1e-6)
    GG=Gm@CX; UU=Um@CX; vg=np.sum(GG*Gm,axis=1); gu=np.sum(GG*Um,axis=1)
    mg=Gm@xbar; mu=Um@xbar
    z=mg[:,None]+np.sqrt(np.maximum(vg,0))[:,None]*zz
    s=expit(z); ds=s*(1-s)
    f=(z*s)@ww; fp=(s+z*ds)@ww; fpp=(ds*(2+z*(1-2*s)))@ww
    fppp=(ds*(3*(1-2*s)+z*(1-6*ds)))@ww
    aus=[]
    for gn in 'ug':
        ga,gb,_=ab(gl,gn); asd=np.sqrt(np.sum((ga@CX)*ga,axis=1)+(ga@xbar)**2+1e-8)
        if gn=='u':
            HF=f[:,None]*(ga@xbar)+fp[:,None]*(GG@ga.T)
        else:
            HF=(fp*mu+gu*fpp)[:,None]*(ga@xbar)+fp[:,None]*(UU@ga.T)+(fpp*mu+gu*fppp)[:,None]*(GG@ga.T)
        HP=HF/asd
        # Physical moment match and standardized Adam sketch.
        hnorm=np.linalg.norm(HP,axis=1); bnorm=np.linalg.norm(gb,axis=1)
        bfac=np.sqrt(np.mean(gb**2,axis=0))
        r=np.sum(HP*gb/np.maximum(bfac,1e-5),axis=1) / np.maximum(hnorm,np.quantile(hnorm,.18))
        r0=np.sum(HP*gb,axis=1)/np.maximum(hnorm,1e-8)/np.maximum(bnorm,1e-8)
        aus.append((r,r0,hnorm))
    return Dmz,aus

brdg={}
for gl in range(8,20):
    dnorm,ags=local_grad_sketch(gl)
    gram_br=dnorm@dnorm.T
    rs_br=np.array([at[j] for at in ags for j in [0,1]]).T
    bb_br=dnorm@rs_br
    brdg[gl]=(gram_br,bb_br,rs_br)
    est_br=la.solve(gram_br+1.5*Ieye,bb_br,assume_a='pos')
    print(gl,'row sim',round(cc(rs_br[:,0],rs_br[:,2]),3), 'axis',np.round([cc(vs13,c) for c in bb_br.T],3),'inverse',np.round([cc(vs13,c) for c in est_br.T],3), 'consistency',round(cc(est_br[:,0],est_br[:,2]),3))