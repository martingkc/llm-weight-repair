def ctxcov_safe(gl,cval=.65):
    return gauss_base(min(gl,20),cval) if gl>20 else gauss_base(gl,cval)

# Rotation kernel for a distribution of relative offsets (not a token sequence).
rope_offsets=np.unique(np.rint(np.r_[np.arange(0,65), np.arange(68,257,4), np.arange(272,513,16)]).astype(int))
rope_width=np.r_[np.diff(rope_offsets),16.]; rope_scores=rope_offsets[:,None]*(1.e6**(-np.arange(32)/32.))[None,:]
rope_sin,rope_cos=np.sin(rope_scores),np.cos(rope_scores)

# Independent Gaussian key/value population: exponential tilt within a RoPE-distance bin.
# Analytic projected mean difference for attention output, split by adapted factors.
def att_weight_contract(gl,mw,oo,cval,stemp):
    bx=(mw*Bmean[gl]+oo*outlier).astype('f')
    CV=ctxcov_safe(gl,cval).astype('f')*max(.22,1-bx@bx/D)
    sn=ln(gl,'a'); mu=sn*bx; Cn=sn[:,None]*CV*sn
    Q0b,K0b,V0b=[w(gl,mv) for mv in 'qkv']; Opb=w(gl,'o')
    Q1b,K1b,V1b=[bb+deltaW(gl,mv) for bb,mv in zip([Q0b,K0b,V0b],'qkv')]
    dmats=[]
    for QQ,KK,VV in [(Q0b,K0b,V0b),(Q1b,K1b,V1b)]:
        qm=QQ@mu+bias(gl,'q');km=KK@mu+bias(gl,'k');vm=VV@mu+bias(gl,'v')
        vc=VV@Cn@KK.T; kc=KK@Cn@KK.T
        wm=[];wm_unw=[]
        for hi in range(14):
            jq=slice(hi*64,(hi+1)*64); jk=slice((hi//7)*64,(hi//7+1)*64)
            ma,mb=qm[jq][:32],qm[jq][32:]
            Z=np.concatenate([rope_cos*ma-rope_sin*mb,rope_sin*ma+rope_cos*mb],axis=1)/8
            logits=Z@km[jk]+.5*np.einsum('ij,ij->i',Z@kc[jk,jk],Z)
            # Average over plausible receptive-field widths and temper the mean-logit asymptotic.
            zp=[]
            for od in [24,80,224,512]:
                mk=rope_offsets<=od
                alpha=logits[mk]*stemp; ww0=rope_width[mk]*np.exp(alpha-alpha.max())
                zp.append((ww0@Z[mk])/ww0.sum())
            zavg=np.mean(zp,axis=0)
            wm.append(vc[jk,jk]@zavg);wm_unw.append(vc[jk,jk]@Z.mean(axis=0))
        dmats.append((repeatkv(vm),np.concatenate(wm),np.concatenate(wm_unw)))
    (va0,tv0,tu0),(va1,tv1,tu1)=dmats
    return np.stack([lrprod(gl,'o',tv1), Opb@(tv1-tv0), lrprod(gl,'o',va1), Opb@(va1-va0),lrprod(gl,'o',tu1)],axis=1)

ro_diffs={}
for gl in range(8,24):
    tr_d0=att_weight_contract(gl,.4,12.,.7,.38)
    tr_d1=att_weight_contract(gl,.0,8.,.5,.65)
    ro_diffs[gl]=.55*tr_d0+.45*tr_d1
    print(gl,np.round([cc(vs13,bb) for bb in ro_diffs[gl].T],2),'reliable',np.round([cc(bb,cc0) for bb,cc0 in zip(tr_d0.T,tr_d1.T)],2),'norm',np.round(np.linalg.norm(ro_diffs[gl],axis=0),3))