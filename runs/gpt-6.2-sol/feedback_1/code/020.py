# First-order Gaussian exponential-tilt attention moment, using only head weights and published biases.
# The moment is evaluated on a symbolic zero-mean prior, not on token inputs.
def rbar(frequency=1e6,sp=96):
    f=frequency**(-(np.arange(32,dtype='f')/32)); pha=np.arange(sp,dtype='f')[:,None]*f
    cs=np.mean(np.cos(pha),axis=0); sn=np.mean(np.sin(pha),axis=0)
    return np.block([[np.diag(cs),-np.diag(sn)],[np.diag(sn),np.diag(cs)]])
rb=rbar(1e6,112)
def actcov(l,c0=.4):
    CC=sum(basec[j] for j in range(max(0,l-6),l)); CC*=D/np.trace(CC)
    CC=c0*CC+Ieye*(1-c0)
    CC-=outlier[:,None]*CC[490,:]; CC-=CC[:,490,None]*outlier;
    r=ln(l,'a'); return (r[:,None]*CC)*r
Ac={}; Ap={}; Av=[]
for l in range(7,19):
    o=w(l,'o'); q=w(l,'q'); v=w(l,'v'); k=w(l,'k'); dq,dv,dk=(deltaW(l,x) for x in 'qvk')
    bq=bias(l,'q'); bv=repeatkv(bias(l,'v')); C=actcov(l); 
    cross=(v@C)@k.T; cross2=((v+dv)@C)@(k+dk).T
    VQ0=np.zeros(D,np.float32); VQ1=np.zeros(D,np.float32); JQ=[]
    for h in range(14):
        sl=slice(h*64,(h+1)*64); sk=slice((h//7)*64,(h//7+1)*64)
        ck=cross[sk,sk]@rb.T/8.; cka=cross2[sk,sk]@rb.T/8.; normf=max(1,np.linalg.norm(ck,2))
        VQ0[sl]=ck@bq[sl]/normf
        VQ1[sl]=cka@bq[sl]/normf
        JQ.append(ck@(q[sl]*ln(l,'a'))/normf)
    Ac[l]=[o@(VQ1-VQ0),lrprod(l,'o',VQ1), lrprod(l,'o',bv), VQ0]
    Ap[l]=o@np.vstack(JQ)*np.sqrt(D)/rnorms[l]
    t=ctx[12]; sd=[-cos(x,t) for x in Ac[l][:3]]
    print(l,'QK effect norm',np.round([la.norm(x) for x in [o@VQ0,o@(VQ1-VQ0),lrprod(l,'o',VQ1),lrprod(l,'o',bv)]],3),'anti align',np.round(sd,2),'Vtilt/Vb',round(cos(VQ0,bv),2), 'Jac sensitivity rel',round(la.norm(Ap[l])/la.norm(Ja[l]),2))
aQ={}
for a,b in [(7,12),(7,15),(9,16),(11,17)]:
    gq=-sum(Ac[l][0]+Ac[l][1] for l in range(a,b));
    qtarget=psmooth(*ctxq[12],gq,.6)
    print(a,b,'cos main',round(cos(gq,vs),2),round(cos(qtarget,vs),2),'led',round(cos(qtarget,ctx[12]),2),'Q read',read(qtarget,7,eng), la.norm(gq)); aQ[a,b]=(un(gq),qtarget)
