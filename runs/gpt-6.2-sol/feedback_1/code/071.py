def swiglu_weight_mom(gl,Cx,xm,adapt=False,out=0.0):
    '''Analytic Gaussian SwiGLU contractions (moments and coefficients of first two Hermite orders).'''
    gg0=w(gl,'g')
    uu0=w(gl,'u')
    dow0=w(gl,'d')
    if adapt:
        gg0=gg0+deltaW(gl,'g'); uu0=uu0+deltaW(gl,'u'); dow0=dow0+deltaW(gl,'d')
    rr0=ln(gl,'m').copy()
    gg0=gg0*rr0; uu0=uu0*rr0
    mg0=gg0@xm; mu0=uu0@xm
    ggC=gg0@Cx
    gvar0=np.einsum('ij,ij->i',ggC,gg0)
    gucov0=np.einsum('ij,ij->i',ggC,uu0)
    zq0=mg0[:,None]+np.sqrt(gvar0)[:,None]*zz[None,:]
    sq0=expit(zq0); ds0=sq0*(1-sq0)
    m0=(zq0*sq0)@ww
    m1=(sq0+zq0*ds0)@ww
    m2=(ds0*(2+zq0*(1-2*sq0)))@ww
    m3=(ds0*(3*(1-2*sq0)+zq0*(1-6*ds0)))@ww
    ef0=m0*mu0+m1*gucov0
    fa0=m1*mu0+m2*gucov0
    ca0=m2*mu0+m3*gucov0
    jn0=gg0*fa0[:,None]+uu0*m0[:,None]
    ha0=gg0*ca0[:,None]+uu0*m1[:,None]
    hb0=gg0*m1[:,None]
    return ef0,jn0,ha0,hb0,gg0,uu0,dow0


def gainmatch(gl,muw=0.55,out=0.0,ce=0.12,gpolicy=0.62,topkeep=1450,wwriter=0.0,colp=vvb0,cp=0.0):
    '''Normal equations for matching the adapter's signed change in linear-response moments, in normalized residual units.'''
    xb=(muw*Bmean[gl]+out*outlier).astype('f')
    Cx=ctxcov_safe(gl,gpolicy).astype('f')*max(.30,1.-(xb@xb)/D)
    ff0,jn0,ha0,hb0,gg0,uu0,dn0 = swiglu_weight_mom(gl,Cx,xb,False)
    ff1,jn1,_,_,_,_,dn1 = swiglu_weight_mom(gl,Cx,xb,True)
    # Project perturbations away from the reference radial component.
    R0=Ieye-np.outer(xb,xb)/D
    Cxn=R0@Cx@R0
    Csel=R0@medcov@R0
    Cxn=(1-ce)*Cxn+ce*(np.trace(Cxn)/np.trace(Csel))*Csel
    bb0=-np.stack([dn0@(jn1-jn0),(dn1-dn0)@jn1],axis=-1).astype('f')
    if wwriter:
        Q0,e0=lpc(max(0,gl-1),min(gl+3,24),'do',.85,26)
        ep0=(e0/e0[-1])**.55
        Aout=(wwriter*ep0)[:,None]**.5*Q0.T
    else:
        Aout=np.zeros((0,D),dtype='f')
    if cp:
        Aout=np.concatenate((Aout, np.sqrt(cp)*un(colp)[None,:]),axis=0)
    Adn=Aout@dn0
    dnP=dn0+Aout.T@Adn
    ggC=gg0@Cxn
    uuC=uu0@Cxn
    ggs=np.einsum('ij,ij->i',ggC,gg0)
    uus=np.einsum('ij,ij->i',uuC,uu0)
    gus=np.einsum('ij,ij->i',ggC,uu0)
    dsqs=np.einsum('ij,ij->j',dn0,dnP)
    contributions=[]
    for j0 in range(2):
        # No token activations: contract the tensor D^T P delta-J C against each base g,u weight row.
        yy0=bb0[:,:,j0]
        V0=(dnP.T@yy0)
        xp0=np.einsum('ij,ij->i',V0,ggC)
        yp0=np.einsum('ij,ij->i',V0,uuC)
        contributions.append(np.stack([ha0.T@xp0,hb0.T@yp0],axis=1).sum(axis=1))
    if topkeep>0:
        imp0=dsqs*(ggs*np.einsum('ij,ij->i',ha0,ha0)+uus*np.einsum('ij,ij->i',hb0,hb0))
        it0=np.argpartition(imp0,-topkeep)[-topkeep:]
        ds=dn0[:,it0].T@dnP[:,it0]
        haa=ha0[it0]; hbb=hb0[it0]; ga0=gg0[it0]; ua0=uu0[it0]
        Hga=(ds*(ggC[it0]@ga0.T))@haa
        Hgb=(ds*(ggC[it0]@ua0.T))@hbb
        Huu=(ds*(uuC[it0]@ua0.T))@hbb
        Hg0=haa.T@(Hga+Hgb)
        Hu0=hbb.T@Huu+ Hgb.T@haa
        dsqs[it0]=0.
        H0=Hg0+Hu0
    else:
        H0=np.zeros((D,D),dtype='f')
    mxy=ha0.T@((dsqs*gus)[:,None]*hb0)
    H0+=ha0.T@((dsqs*ggs)[:,None]*ha0)+hb0.T@((dsqs*uus)[:,None]*hb0)+mxy+mxy.T
    return R0@H0@R0,R0@np.stack(contributions,axis=1)

t_start=time.time()
Xg0,Xr0=gainmatch(13,wwriter=1.8,cp=2.0)
print('elapsed', time.time()-t_start,'trace',np.trace(Xg0)/D)
rr0=la.solve(Xg0+np.trace(Xg0)/D*.65*Ieye,Xr0,assume_a='pos'); print([round(float(cos(z0,vvb0)),3) for z0 in rr0.T], [la.norm(z0) for z0 in rr0.T])