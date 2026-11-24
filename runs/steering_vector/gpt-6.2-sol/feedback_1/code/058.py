def get_tilt_drift(gl):
    if gl<=18:
        ff1s=[BE[gq][gl] for gq in [.65,1.,-.65]]
        d0m=np.mean([xj[0] for xj in ff1s],axis=0)
        d1m=np.mean([xj[1] for xj in ff1s],axis=0)
    else:
        d1m=(np.zeros(D)); fn=ln(gl,'m'); CX=ctxcov_safe(gl,.62).astype('f'); Gg=w(gl,'g')*fn; Ug=w(gl,'u')*fn; win=Gg@CX
        fm=gaussian_feat(np.zeros(Gg.shape[0]),np.zeros(Gg.shape[0]),np.sum(win*Gg,axis=1),np.sum(win*Ug,axis=1)); d0m=lrprod(gl,'d',fm)
        d1m=w(gl,'d')@(.5*(np.sum((deltaW(gl,'g')*fn)@(CX)*Ug,axis=1)+np.sum(win*(deltaW(gl,'u')*fn),axis=1)))
    aw=ro_diffs[gl]; typv0=aw[:,0]; typv1=aw[:,1]; typb=aw[:,2]; typv2=aw[:,3]
    return np.stack([d0m,d1m,typv0,typv1,typb,typv2],axis=1)

expDrift={gl:get_tilt_drift(gl) for gl in range(8,24)}
# Preconditioned equation for multi-layer cancellation of the first moment across representative depths.
def cancel_agg(L0,drif={},ap=.26,mp=.6,ste=(.7,1.,1.15,.35,.12,.23),lengths=(13,15,17,20,24)):
    exv={}
    ww2=np.array(ste)
    for gl,Bg in drif.items():
        if gl<10 or gl>16:
            loenergy=(1/(1+.25*(gl-13)**2))**.44
        else:
            loenergy=1.
        vlevels=np.linalg.norm(Bg,axis=0); blim=.012*np.array([.93,1.2,1.5,1.0,.3,.75]); scales=np.maximum(vlevels,blim)
        # Robust weighting of contraction scales against the incidental value bias.
        dm0=Bg@(ww2/np.sqrt(scales))*loenergy
        exv[gl]=dm0
    ts={tl:mat_rng(L0,tl,ap,mp) for tl in lengths}
    AA0=np.eye(D)*.9; bvec=0.
    for ti,tl in enumerate(lengths):
        mx=ts[tl]; mixv=np.zeros(D)
        for gl,phi in exv.items():
            if gl>=tl: continue
            tg=mat_rng(gl+1,tl,ap,mp)
            mixv-=tg@phi
        hh=.6*Ieye+.4*EH
        AA0+=mx.T@hh@mx
        bvec+=mx.T@hh@mixv
    return la.solve(AA0,bvec,assume_a='pos')

hagg={}
for sw in [(1,1,1,1,1,1),(.7,1,1.15,.35,.12,.23),(.3,1.2,1.5,.3,0,.4)]:
 for ends in [(13,15,17,20,24),(14,15,17,18,20),(15,17,20,22,24)]:
    hrz=cancel_agg(13,expDrift,ste=sw,lengths=ends)
    aa0=(cbk.T@un(hrz)); hagg[(sw,ends)]=hrz
    print(sw,ends, np.round([np.linalg.norm(hrz),cc(vs13,hrz),cc(vs13,cbk@(cbk.T@hrz)),cc(hrz,brd_reps[13])],3),np.round(aa0[:8],3))