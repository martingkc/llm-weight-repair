note('Best so far context12_signed_mlp (norm 2.2): med .3533/general .3183. It adds a shrinkage-filtered anti-MLP drift from analytic SwiGLU moments using a covariance proxy (base residual writer Gram matrices). Transported LoRA covariance variant mapped_13 .3146/.2546. The signed drift improves both metrics, confirming useful information beyond semantic anchoring.')
Tomo={}; Ttail={}
for l in range(5,20):
    R=ln(l,'m'); Ra=ln(l,'a'); d=w(l,'d'); o=w(l,'o'); vr=w(l,'v')*Ra
    en=t=ctx[12] if 7<l<16 else ctx[l] if l<20 else ctx[12]
    af,bf=phid[l]; Gd=(d.T@t)
    vt=(o.T@t).reshape(2,7,64).sum(1).ravel()
    ts={}
    for m,gm,rr in [('u',Gd*af,R),('g',Gd*bf,R),('v',vt,Ra),('o',t,np.ones(D)),('d',t,np.ones(I))]:
        U1,s1,V1=ls[l,m]
        A,B,s=ab(l,m)
        x=V1.T@(s1*(U1.T@gm)); xt=tail(l,m,.25).T@(B.T@gm)
        ts[m]=(x,xt)
    xx=[ts[m][0] for m in 'gvu']; xt=[ts[m][1] for m in 'gvu']; xo=ts['o'][0]; xd=ts['d'][0]
    scoreo=-cos(xo,repeatkv(bias(l,'v'))); scored=-cos(xd,feat[l]);
    xv=-un(np.concatenate([(vr.T@xo.reshape(2,7,64).sum(axis=1).ravel())[None,:],np.array(xx)],0),1)
    xt=-un(np.stack(xt),1);
    Tomo[l]=(xv,ts); Ttail[l]=xt
    print(l,'d/o sign',np.round([scored,scoreo],2),'c',np.round([xv[i]@xv[j] for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],2),'synth', np.round([cos(q,cmu[l-1]) for q in xv],2),'bn',np.round(np.linalg.norm(xv,axis=1),2))
print('pooled cosines')
for l in range(8,16):
    Y=np.stack([Tomo[s][0] for s in range(l-2,l+3)]); a=un(Y.sum(0),1)
    z=un(sum(cmu[j] for j in range(l))); iso=unit=np.zeros_like(z); iso[490]=1;
    print(l,np.round(a@a.T,2), 'to static prior',np.round(a@z,2),'to outlier',np.round(a@iso,2),'tail',np.round(np.array([cos(ai,ti) for ai,ti in zip(a[1:],un(np.stack([Ttail[s] for s in range(l-2,l+3)]).sum(0),1))]),2))