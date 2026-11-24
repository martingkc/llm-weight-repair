def att_marginal_drift(gl,mw=.45,kshift=0.,qshift=0.,qv=.07,cw=.62,qknow=.0):
    # Average moments of an exponentially tilted Gaussian Q,K,V distribution;
    # marginalize some query variance, rather than substituting only the query mean.
    xq0=(mw*Bmean[gl]+qshift*outlier).astype('f')
    xk0=(mw*Bmean[gl]+kshift*outlier).astype('f')
    if qknow:
        xk0=xk0+qknow*un(cue_ref).astype('f')
    C0=ctxcov_safe(gl,cw).astype('f')
    Cq=C0*max(.25,1.-(xq0@xq0)/D)
    Ck=C0*max(.25,1.-(xk0@xk0)/D)
    S0=ln(gl,'a')
    M0=[w(gl,proj)*S0 for proj in 'qkv']
    M1=[m0+deltaW(gl,cr0)*S0 for m0,cr0 in zip(M0,'qkv')]
    b0=[bias(gl,proj) for proj in 'qkv']
    nsel0=np.unique(np.r_[np.arange(16),np.arange(16,len(rope_offsets),3)])
    rpos=rope_offsets[nsel0]
    rw0=rope_width[nsel0].copy(); rw0[16:]*=3
    cr=np.array(rope_cos[nsel0,:]); sr=np.array(rope_sin[nsel0,:]); ns0=len(cr)
    # RoPE pullback operator, per summarized distance bucket.
    RR=np.zeros((ns0,64,64),dtype='f'); rj=np.arange(32)
    RR[:,rj,rj]=cr;RR[:,32+rj,32+rj]=cr;RR[:,rj,32+rj]=-sr; RR[:,32+rj,rj+32]=sr
    yys=[]
    for md0 in [M0, M1]:
        Q0,K0,V0=md0
        qm=Q0@xq0+b0[0]; km=K0@xk0+b0[1]; vm=V0@xk0+b0[2]
        Kv0=K0@Ck@K0.T; Vc0=V0@Ck@K0.T
        parts=[]; varparts=[]; taum=[]
        for hd0 in range(14):
            sl0=slice(hd0*64,(hd0+1)*64); sv0=slice(hd0//7*64,(hd0//7+1)*64)
            qq=Q0[sl0,:]; Km=Kv0[sv0,sv0]
            Cqa=qq@Cq@qq.T/64
            mZ=(RR@qm[sl0,None])[:,:,0]/8
            ev1=max(1.e-8, float(la.norm(Cqa,ord=2)*la.norm(Km,ord=2)))
            qv0=min(qv,.62/ev1)
            Sqa=qv0*Cqa
            Sza=(RR@Sqa)@RR.transpose(0,2,1)
            Sm=Km@mZ.T+km[sv0,None]
            r0=Sza@Sm.T[:,:,None]
            # Solve a batch of small analytic moment normalizations, not token-dependent attention.
            Lmm=(np.eye(64,dtype='f')[None,:,:]-Sza@Km).astype('f')
            exZ=np.linalg.solve(Lmm,r0)[:,:,0]
            dz=(np.einsum('ni,in->n',exZ,Sm))*.5
            _,ld=np.linalg.slogdet(Lmm)
            logw0=np.einsum('ij,ji->i',mZ,km[sv0,None]+.5*(Km@mZ.T))
            logwc=logw0+dz-.5*ld
            YZ=[]; Y0Z=[]
            for cutoff in [24,80,224,512]:
                mmk=rpos<=cutoff; wl=logwc[mmk]; wlm=logw0[mmk]
                wt0=np.exp(wl-wl.max())*rw0[mmk]
                wm0=np.exp(wlm-wlm.max())*rw0[mmk]
                YZ.append((mZ[mmk,:]+exZ[mmk,:]).T@wt0/np.sum(wt0)); Y0Z.append(mZ[mmk,:].T@wm0/np.sum(wm0))
            pt0=Vc0[sv0,sv0]@np.mean(YZ,axis=0)
            p00=Vc0[sv0,sv0]@np.mean(Y0Z,axis=0)
            parts.append(pt0); varparts.append(pt0-p00);taum.append(qv0)
        yys.append([np.concatenate(parts),repeatkv(vm),np.concatenate(varparts),taum])
    OO=w(gl,'o'); (yb,vb,rb0,tb0),(ya,va,ra0,ta0)=yys
    return np.stack([lrprod(gl,'o',ya), OO@(ya-yb),lrprod(gl,'o',va),OO@(va-vb),lrprod(gl,'o',ra0),OO@(ra0-rb0)],axis=1)
_tt0=time.time(); Us0=att_marginal_drift(13)
print('elapsed',time.time()-_tt0, 'norms',np.round(np.linalg.norm(Us0,axis=0),4),'cos',[round(float(cos(z0,vvb0)),3) for z0 in Us0.T]); print('agreement',cos(Us0[:,0],expDrift[13][:,2]),cos(Us0[:,1],expDrift[13][:,3]))