# Analytic Gaussian attention prior; RPM and two token-free populations of query/key moments.
# Each entry is a contraction of weights and Gaussian parameters, with no sequence of model inputs.
def att_variant(gl,mw=.50,kshift=0.,qshift=0.,qv=.09,cw=.62,cwp=.10,medq=0.,kcue=cue_ref):
    xq0=(mw*Bmean[gl]+qshift*outlier).astype('f')
    xk0=(mw*Bmean[gl]+kshift*outlier+medq*un(kcue)).astype('f')
    C0=ctxcov_safe(gl,cw).astype('f')
    Cq=C0*max(.25,1.-(xq0@xq0)/D)
    Cm=(1-cwp)*C0+cwp*medcov
    Ck=Cm*max(.25,1.-(xk0@xk0)/D)
    S0=ln(gl,'a')
    M0=[w(gl,pr0)*S0 for pr0 in 'qkv']; M1=[m0+deltaW(gl,pr0)*S0 for m0,pr0 in zip(M0,'qkv')]
    b0=[bias(gl,pr0) for pr0 in 'qkv']
    nsel0=np.unique(np.r_[np.arange(20),np.arange(20,len(rope_offsets),2)])
    rpos=rope_offsets[nsel0]
    rw0=rope_width[nsel0].copy(); rw0[20:]*=2
    cr=np.array(rope_cos[nsel0,:]); sr=np.array(rope_sin[nsel0,:]); ns0=len(cr)
    RR=np.zeros((ns0,64,64),dtype='f'); rj=np.arange(32)
    RR[:,rj,rj]=cr; RR[:,32+rj,32+rj]=cr; RR[:,rj,32+rj]=-sr; RR[:,32+rj,rj]=sr
    yys=[]
    for md0 in [M0,M1]:
        Q0,K0,V0=md0; qm=Q0@xq0+b0[0]; km=K0@xk0+b0[1]; vm=V0@xk0+b0[2]
        Kv0=K0@Ck@K0.T; Vc0=V0@Ck@K0.T
        parts=[]; varparts=[]
        for hd0 in range(14):
            sl0=slice(hd0*64,(hd0+1)*64); sv0=slice(hd0//7*64,(hd0//7+1)*64)
            qq=Q0[sl0,:]; Km=Kv0[sv0,sv0]
            Cqa=qq@Cq@qq.T/64
            mZ=(RR@qm[sl0,None])[:,:,0]/8
            ev1=max(1.e-8, float(la.norm(Cqa,ord=2)*la.norm(Km,ord=2)))
            qv0=min(qv,.65/ev1)
            Sm=Km@mZ.T+km[sv0,None]
            Sza=(RR@(qv0*Cqa))@RR.transpose(0,2,1)
            Lmm=(np.eye(64,dtype='f')[None,:,:]-Sza@Km).astype('f')
            exZ=np.linalg.solve(Lmm,Sza@Sm.T[:,:,None])[:,:,0]
            dz=np.einsum('ni,in->n',exZ,Sm)*.5
            with np.errstate(all='ignore'):
                _,ld=np.linalg.slogdet(Lmm.astype('d'))
            if not np.all(np.isfinite(ld)):
                raise ValueError('non-finite moment partition')
            lo0=np.einsum('ij,ji->i',mZ,km[sv0,None]+.5*(Km@mZ.T))
            lo1=lo0+dz-.5*ld
            Zmom=[]; Zcor=[]
            for cutoff in [24,80,224,512]:
                mmk=rpos<=cutoff
                # Small temperature damping marginalizes uncertainty about query position distributions.
                wl=.72*lo1[mmk]; wlm=.72*lo0[mmk]
                wt0=np.exp(wl-wl.max())*rw0[mmk]; wm0=np.exp(wlm-wlm.max())*rw0[mmk]
                Zcor.append((mZ[mmk,:]+exZ[mmk,:]).T@wt0/np.sum(wt0)); Zmom.append(mZ[mmk,:].T@wm0/np.sum(wm0))
            pt0=Vc0[sv0,sv0]@np.mean(Zcor,axis=0); p00=Vc0[sv0,sv0]@np.mean(Zmom,axis=0)
            parts.append(pt0); varparts.append(pt0-p00)
        yys.append([np.concatenate(parts),repeatkv(vm),np.concatenate(varparts)])
    OO=w(gl,'o'); (yb,vb,rb0),(ya,va,ra0)=yys
    return np.stack([lrprod(gl,'o',ya), OO@(ya-yb),lrprod(gl,'o',va),OO@(va-vb),lrprod(gl,'o',ra0),OO@(ra0-rb0)],axis=1)

tas0={}; ti0=time.time()
ctv0=[(.50,0.,0.,.09,.62,0.,0.),(.72,7.,5.,.12,.69,0.,0.),(.42,-7.,-5.,.12,.62,0.,0.),(.48,3.,1.,.09,.68,.18,3.),(.48,3.,1.,.09,.68,.18,-3.)]
for si0,params0 in enumerate(ctv0):
 tas0[si0]={gl:att_variant(gl,*params0) for gl in range(8,24)}
print('time',time.time()-ti0)
for si0 in range(5):
 rr=TAS=np.stack([sum(mat_rng(gl+1,13,.23,.55)@tas0[si0][gl][:,cc0]/(la.norm(tas0[si0][gl][:,cc0])+.02)**.5 for gl in range(8,19)) for cc0 in range(6)],1)
 print(si0, np.round(la.norm(rr,axis=0),3), 'corr',np.round(un(vvb0)@rr/la.norm(rr,axis=0),3))
print('modulo q contrib',cos(tas0[0][13][:,0],expDrift[13][:,2]),cos(tas0[0][13][:,1],expDrift[13][:,3]))