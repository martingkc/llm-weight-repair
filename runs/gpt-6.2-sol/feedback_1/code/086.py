def writer_jdiff(gl,mw=.44,cw=.60):
    x0=(Bmean[gl]*mw).astype('f')
    CF0=ctxcov_safe(gl,cw).astype('f')
    CF0*=1.-(x0@x0)/D
    Ra0=Ieye-np.outer(x0,x0)/D
    mf0,jn0,_,_,_,_,D0=swiglu_weight_mom(gl,CF0,x0)
    mf1,jn1,_,_,_,_,D1=swiglu_weight_mom(gl,CF0,x0,adapt=True)
    DM0=((D1-D0)@jn0)@Ra0
    JM1=(D1@(jn1-jn0))@Ra0
    atnorm=ln(gl,'a'); Ov0=w(gl,'o'); do0=deltaW(gl,'o');
    v0=w(gl,'v')*atnorm; dv0=deltaW(gl,'v')*atnorm
    q0=w(gl,'q')*atnorm; dq0=deltaW(gl,'q')*atnorm
    k0=w(gl,'k')*atnorm; dk0=deltaW(gl,'k')*atnorm
    AO0=(do0@repeatkv(v0))@Ra0
    AV0=((Ov0+do0)@repeatkv(dv0))@Ra0
    TQA0=np.zeros((D,D),dtype='f'); TQA1=np.zeros_like(TQA0)
    for ia0,(Q0,K0,V0,TQ0) in enumerate(zip([q0,q0+dq0],[k0,k0+dk0],[v0,v0+dv0],[TQA0,TQA1])):
        Ckv=V0@CF0@K0.T
        for j0 in range(14):
            h0=slice(j0*64,(j0+1)*64); h1=slice(j0//7*64,(j0//7+1)*64)
            # Average RoPE at a range of distances, without forming attention matrices.
            TQ0[h0,:]=(Ckv[h1,h1]@rb.T)@Q0[h0,:]/8.
    ARO0=(do0@TQA0)@Ra0
    Atilt0=((Ov0+do0)@(TQA1-TQA0))@Ra0
    return [DM0,JM1,AO0,AV0,ARO0,Atilt0]
JdifGold={gl:writer_jdiff(gl) for gl in range(8,19)}
print('diff component norms', np.round(np.array([[la.norm(z0) for z0 in JdifGold[gl]] for gl in range(8,19)]),2))
Qs0,Qd0,Es0=proj_pow(ccH,returnQ=True,top0=40)

Jsum0=np.zeros((6,D,D),dtype='f'); DCb=ctxcov_safe(13,.62)
for gl in range(8,19):
    Tpre=mat_rng(13,gl,.22,.5)
    Tout=mat_rng(gl+1,13,.22,.5)
    wgl=(1.+((gl-12.5)/3.5)**2)**-.70
    for ic,rm0 in enumerate(JdifGold[gl]):
        Jsum0[ic]+=wgl*(Tout@rm0@Tpre)/(la.norm(rm0)+.02)**.75
for kl0 in [0,1,2,3,4,5,6]:
 ns0=Jsum0[kl0] if kl0<6 else np.sum(Jsum0,axis=0)
 dn3=Qd0.T@ns0; sn0=la.svdvals(dn3)
 nn0=la.norm(dn3); nx0=la.norm(ns0)
 print(kl0,'norm fraction',round(float(nn0/nx0),2),'top sv',np.round(sn0[:6]/nn0,3),'drift pullback cos', np.round([cos(ns0.T@x0,vvb0) for x0 in [vvb0,phys,ctx[12]]],3),'drift pushforward',np.round([cos(ns0@x0,vvb0) for x0 in [Bmean[13],cmu[13],np.mean(E[eng[:30000]],0),vvb0]],2))