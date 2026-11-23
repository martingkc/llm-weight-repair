def gain_pool(L0,layers0,ce,weight0=2.,out=0.,mw=.55):
    HS=np.zeros((D,D),dtype='f'); RS=np.zeros((D,2),dtype='f'); Svars=[]
    for gl in layers0:
        g0,r0=gainmatch(gl,muw=mw,out=out,ce=ce,wwriter=weight0,cp=weight0,topkeep=1450)
        rmsf=np.sqrt(D)/rnorms[gl]
        T0=(mat_rng(L0,gl,.26,.60)*rmsf).astype('f')
        weg=np.exp(-((gl-L0+.9)/5.)**2)
        q0=weg/(np.trace(g0)/D)
        Hm0=T0.T@g0@T0; Rm0=T0.T@r0
        HS+=q0*Hm0;RS+=q0*Rm0
        Svars.append([g0,r0])
    xs0=la.solve(HS+.8*np.trace(HS)/D*Ieye,RS,assume_a='pos')
    return xs0,Svars,HS,RS
Xlist0={ce0:gain_pool(13,range(8,19),ce0) for ce0 in [0.,.22,.65]}
for ce0,(xs0,Svars,HS,RS) in Xlist0.items(): print(ce0, 'magnitudes',la.norm(xs0,axis=0), 'cos to best/gate/drift',[[round(float(cos(z0,x0)),3) for z0 in xs0.T] for x0 in [vvb0,brd_reg,phys,k_comp]])
print('agreement', [round(float(cos(Xlist0[0.][0][:,0],Xlist0[ce0][0][:,0])),3) for ce0 in [.22,.65]])