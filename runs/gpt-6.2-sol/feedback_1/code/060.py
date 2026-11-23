def moment_lin(gl,muw=.46,out=16., cval=.58,kw=.23):
    # First derivatives of the analytically integrated means with respect to a constant residual translation.
    xn=(muw*Bmean[gl]+out*outlier).astype('f'); CF=ctxcov_safe(gl,cval).astype('f')*max(.18,1-xn@xn/D)
    Pn=(Ieye-np.outer(xn,xn)/D-CF/D)*(np.sqrt(D)/rnorms[gl]); Pn=Pn.astype('f')
    r_m=ln(gl,'m'); g_m=w(gl,'g')*r_m; u_m=w(gl,'u')*r_m; dm=w(gl,'d')
    c_g=g_m@CF; sgm=np.sqrt(np.maximum(np.sum(c_g*g_m,axis=1),0)); gc_u=np.sum(c_g*u_m,axis=1)
    ggmu=g_m@xn; uumu=u_m@xn
    z0=ggmu[:,None]+sgm[:,None]*zz; ss0=expit(z0); d0=ss0*(1-ss0)
    f0=(z0*ss0)@ww; fp=(ss0+z0*d0)@ww; fpp=(d0*(2+z0*(1-2*ss0)))@ww
    Jm0=dm@((fp*uumu+fpp*gc_u)[:,None]*g_m+f0[:,None]*u_m)@Pn
    r_a=ln(gl,'a'); qa=w(gl,'q')*r_a; ka=w(gl,'k')*r_a; va=w(gl,'v')*r_a; oa=w(gl,'o')
    qmu=qa@xn+bias(gl,'q'); kmu=ka@xn+bias(gl,'k'); covk=(ka@CF)@ka.T; vkc=(va@CF)@ka.T
    ja0=oa@np.tile(np.repeat(va.reshape(2,64,D),7,axis=0).reshape(896,D),(1,1))
    ja1=np.empty((D,D),dtype=np.float32)
    for hi in range(14):
        jq=slice(hi*64,(hi+1)*64); jk=slice((hi//7)*64,(hi//7+1)*64)
        qi=qmu[jq]; qq=qa[jq]; Kk=ka[jk]
        zA=rope_cos*qi[:32]-rope_sin*qi[32:]; zB=rope_sin*qi[:32]+rope_cos*qi[32:]; Z=np.concatenate([zA,zB],axis=1)/8
        kF=covk[jk,jk]; kvF=vkc[jk,jk]; conf=kmu[jk]+Z@kF
        logs=Z@kmu[jk]+.5*np.einsum('ij,ij->i',Z@kF,Z)
        # Expected derivative of the normalized moment kernel, after summing distance bins.
        # Row p of the score derivative uses R_p^T applied to the analytic tilt input.
        ua0=rope_cos*conf[:,:32]+rope_sin*conf[:,32:]; ub0=-rope_sin*conf[:,:32]+rope_cos*conf[:,32:]
        krot=np.concatenate([ua0,ub0],axis=1)/8
        zk1=krot@qq+Z@Kk
        awt=0.
        for od in [28,96,256,512]:
            mk=rope_offsets<=od
            alpha=logs[mk]*.45; wpre=rope_width[mk]*np.exp(alpha-alpha.max()); wt=wpre/wpre.sum()
            sca=wt@rope_cos[mk]; ssa=wt@rope_sin[mk]; zavg=wt@Z[mk];
            ma=(sca[:,None]*qq[:32]-ssa[:,None]*qq[32:])/8; mb=(ssa[:,None]*qq[:32]+sca[:,None]*qq[32:])/8
            awt+=(np.concatenate([ma,mb],axis=0)+.45*(((Z[mk]-zavg).T*wt)@zk1[mk]))*.25
        ja1[jq]=kvF@awt
    Ja0=(ja0+kw*(oa@ja1))@Pn
    return Jm0,Ja0

momentJ={}
for gl in range(8,24):
    jf,jk=moment_lin(gl)
    momentJ[gl]=(jf,jk)
    print(gl, np.round([cc(jf.ravel(),Jm[gl].ravel()),cc(jk.ravel(),Ja[gl].ravel()), np.linalg.norm(jf)/np.linalg.norm(Jm[gl]),np.linalg.norm(jk)/np.linalg.norm(Ja[gl])],3))