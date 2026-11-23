def mat_root(X0,clip0=0.):
    ee0,eq0=la.eigh(.5*(X0+X0.T))
    return ((np.maximum(ee0,clip0)**.5)[None,:]*eq0)@eq0.T

def function_writers(gl,mw=.47,sink=0.0,cw=.64,extra=True):
    xb=(mw*Bmean[gl]+sink*outlier).astype('f')
    Cx=ctxcov_safe(gl,cw).astype('f')*max(.30,1.-(xb@xb)/D)
    mf,jn,ha,hb,g0,u0,d0=swiglu_weight_mom(gl,Cx,xb)
    xc1=g0@Cx; xc2=u0@Cx
    vaG=np.maximum(1e-7,np.einsum('ij,ij->i',xc1,g0))
    vaU=np.maximum(1e-7,np.einsum('ij,ij->i',xc2,u0))
    covGU=np.einsum('ij,ij->i',xc1,u0)
    sqG=vaG**.5
    gx=(g0@xb)[:,None]+sqG[:,None]*zz
    mU=u0@xb
    condU=mU[:,None]+(covGU/sqG)[:,None]*zz
    varU=np.maximum(1e-7,vaU-covGU**2/vaG)
    gu=expit(gx); dgu=gu*(1-gu); f=gx*gu
    fp=gu+gx*dgu; fpp=dgu*(2+gx*(1-2*gu))
    fvar=(f**2*(condU**2+varU[:,None]))@ww-mf**2
    sgu=(fp*condU)@ww
    suu=f@ww
    dstd=np.maximum(0.,fp**2@(ww)*varU+(fp**2*condU**2)@ww-sgu**2)**.5
    ustd=np.maximum(0.,f**2@ww-suu**2)**.5
    af0,bf0,sc0=ab(gl,'d'); F0=af0@jn
    ai0=F0@Cx@F0.T
    vaLin=np.einsum('ij,ij->i',jn@Cx,jn)
    var_rem=np.maximum(.1*fvar,fvar-vaLin)
    ai0+=(af0*var_rem)@af0.T
    amean=af0@mf
    Cd_n=ai0+.35*np.outer(amean,amean)
    da0=(sc0*bf0@amean)
    fd0=sc0*bf0@mat_root(Cd_n)
    av0,bv0,sv0=ab(gl,'o');
    dv0=w(gl,'v')*ln(gl,'a'); bv_bias=bias(gl,'v'); oo0=w(gl,'o')
    vm0=dv0@xb+bv_bias
    vf0=av0@repeatkv(dv0)
    co1=vf0@Cx@vf0.T
    co2=np.zeros_like(co1)
    for hj in range(14):
        op0=av0[:,hj*64:(hj+1)*64]@dv0[hj//7*64:(hj//7+1)*64,:]
        co2+=op0@Cx@op0.T
    ov0=av0@repeatkv(vm0)
    Co_n=.5*co1+.5*co2+.20*np.outer(ov0,ov0)
    fo0=sv0*bv0@mat_root(Co_n)
    oa0=sv0*bv0@ov0
    effects0=[fd0,fo0]
    means0=[da0,oa0]
    if extra:
        dz0 = np.stack([ma0*ln(gl,'m') for ma0 in [ab(gl,mg0)[0] for mg0 in 'gu']])
        for zzj0,(zn0,m0,st0) in enumerate(zip('gu',[sgu,suu],[dstd,ustd])):
            a0,b0,s0=ab(gl,zn0)
            a0=dz0[zzj0]
            tz0=mat_root(a0@Cx@a0.T)
            valm0=(d0*m0)@b0
            vals0=(d0*st0)@b0
            effects0.append(np.concatenate([s0*valm0@tz0,.5*s0*vals0@tz0],axis=1))
            ef_m=(Cx@a0.T)
            ameanU0=(u0@ef_m); ameanG0=(g0@ef_m)
            if zn0=='g':
                sv1_0=(fpp*condU)@ww
                a2_m=sv1_0[:,None]*ameanG0+fp.__matmul__(ww)[:,None]*ameanU0
            else:
                a2_m=(fp@ww)[:,None]*ameanG0
            means0.append(d0@(s0*((b0*a2_m).sum(axis=1)+(m0[:,None]*b0)@(a0@xb))))
        vMa=oo0@repeatkv(np.eye(128,dtype='f'))
        ag0,bg0,sg0=ab(gl,'v');dAv=ag0*ln(gl,'a')
        vd=mat_root(dAv@Cx@dAv.T)
        effects0.append(sg0*(vMa@bg0)@vd)
        means0.append(sg0*(vMa@bg0)@(dAv@xb))
    return effects0,np.stack(means0,axis=1)
fw0={gl:function_writers(gl) for gl in range(8,24)}
print('near physical output norms d o g u v:')
for gl in range(9,18):
 fff0,mmm0=fw0[gl]; print(gl,np.round([la.norm(z0) for z0 in fff0],3),np.round(la.norm(mmm0,axis=0),3),np.round([cos(z0,vvb0) for z0 in mmm0.T],2))
print('compare change of covariance')
for gl in [9,11,12,13,14,15,20,22,23]:
 fff0,mmm0=fw0[gl]; qbb0=lpc(gl,gl+1,'do',1,12)[0]
 print(gl, np.round([la.norm(z0.T@qbb0)/la.norm(z0) for z0 in fff0],2),np.round([la.norm(z0.T@un(vvb0))/la.norm(z0) for z0 in fff0],2))