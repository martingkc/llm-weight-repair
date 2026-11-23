# Static rotary score/value covariance and first-moment differences from weight
# contractions alone. There are no token inputs or attention matrices.
rot_offsets=np.array([0.,1.,2.,4.,8.,16.,32.,64.],np.float32)
rot_fact=(1e6)**(-np.arange(32,dtype=np.float32)/32.0)
rot_w=np.array([.08,.14,.16,.17,.16,.13,.10,.06],np.float32)
def tilt_statistics(ll,using_delta=False, a_v=-.22, cov_v=.06, a_q=-.16, cap=1.5, qk_delta=None):
    ba_q,ba_k,ba_v=[load(pre(ll)+name+'.bias') for name in [qn,kn,vn]]
    mats=[wt(ll,name).copy() for name in [qn,kn,vn,on]]
    if qk_delta is None: qk_delta=using_delta
    if qk_delta:
        for j0,name in enumerate([qn,kn]): mats[j0]+=delta(pre(ll)+name)
    if using_delta:
        for j0,name in enumerate([vn,on],2): mats[j0]+=delta(pre(ll)+name)
    qw,kw,vw,ow=mats
    ga = AN[ll]
    rx,rv = cmread[ll]
    k_dir=ga*rv
    q_meanb=(ba_q+(a_q*np.sqrt(D))*(qw@(ga*rx))).reshape(14,64)
    k_head=(kw*ga[None,:]).reshape(2,64,D).repeat(7,axis=0)
    v_head=(vw*ga[None,:]).reshape(2,64,D).repeat(7,axis=0)
    # Before considering scoring, the context second moment is an isotropic prior
    # plus an anisotropy supported by the signed learned value-read direction.
    k_r=k_head@rv
    v_r=v_head@rv
    vu0=ba_v.reshape(2,64).repeat(7,axis=0)+(a_v*np.sqrt(D))*v_r
    cross_kv=(1.-cov_v)*np.matmul(v_head,k_head.transpose(0,2,1))+(cov_v*D)*v_r[:,:,None]*k_r[:,None,:]
    cov_kk=(1.-cov_v)*np.matmul(k_head,k_head.transpose(0,2,1))+(cov_v*D)*k_r[:,:,None]*k_r[:,None,:]
    qt=rotnom(q_meanb[None,:,:],rot_offsets[:,None,None]*rot_fact[None,None,:])/8.
    mass=(np.einsum('lhj,hjk,lhk->lh',qt,cov_kk,qt)).clip(0)
    gain=1/np.sqrt(1+mass/(cap*cap))
    vv0=(np.einsum('hij,lhj,lh,l->hi',cross_kv,qt,gain,rot_w)).reshape(D)
    return (ow@vu0.ravel(),ow@vv0,vu0.ravel(),vv0,np.sqrt(mass).mean())
TV0={ll:tilt_statistics(ll,False) for ll in range(9,17)}
TV1={ll:tilt_statistics(ll,True) for ll in range(9,17)}
TV2={ll:tilt_statistics(ll,True,qk_delta=False) for ll in range(9,17)}
for ll in range(9,17):
    r0,r1,r2=TV0[ll],TV1[ll],TV2[ll]
    qq=pcwrite([ll]);xs0=unit(-act_sig_grad(ll,on,qq))
    d0,d1,d2=r1[0]-r0[0],r1[1]-r0[1],r1[1]-r2[1]
    print(ll,'O signature',[round(float(unit(x)@xs0),3) for x in [r0[2],r0[3],r0[2]+r0[3]]], 'delta writes',np.round([[la.norm(x),x@qq,x@best] for x in [d0,d1,d2]],4).tolist(),'score SD',round(float(r0[4]),2))