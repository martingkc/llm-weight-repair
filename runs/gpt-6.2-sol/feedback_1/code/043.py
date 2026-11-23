def tail_est(a,power=0.5):
    aa=np.abs(a)*np.sqrt(a.shape[1]); return np.sign(a)*np.maximum(aa-1,0)**power if power else np.sign(a)*(aa>1)

def moment_inputs(gl,cw=.65,muw=.55,big=0.):
    xbar=(muw*Bmean[gl]+big*outlier).astype('f')
    CX=(max(.22,1-xbar@xbar/D))*gauss_base(gl,cw).astype('f')
    ff=ln(gl,'m');fa=ln(gl,'a');Gm=w(gl,'g')*ff;Um=w(gl,'u')*ff
    CG=Gm@CX
    f0=gaussian_feat(Gm@xbar,Um@xbar,np.sum(CG*Gm,axis=1),np.sum(CG*Um,axis=1))
    CNa=fa*(CX*fa[:,None]); Qm=w(gl,'q'); Km=w(gl,'k');Vm=w(gl,'v'); vk=Vm@CNa@Km.T; bQ=Qm@(fa*xbar)+bias(gl,'q')
    fov=(Vm@(fa*xbar)+bias(gl,'v'))
    fvo=repeatkv(fov); fvt=np.concatenate([(vk[(h//7)*64:(h//7+1)*64,(h//7)*64:(h//7+1)*64]@rb.T@bQ[h*64:(h+1)*64])/8 for h in range(14)])
    return {'d':f0, 'o':fvo+0.8*fvt, 't':fvt, 'a':fvo, 'm':xbar*ff,'r':xbar*fa}

tail_results={}
for gl in range(8,21):
 inp=moment_inputs(gl,big=7.)
 vvm=[]
 for mt,iy in [('d','d'),('o','o'),('o','t'),('o','a')]:
    ta,tb,tsc=ab(gl,mt); gx=inp[iy]; tz=tail_est(ta,.5); prog=tb.T@vs13
    vvm.append([cc(prog,ta@gx),cc(prog,tz@gx),cc(ta@gx,tz@gx),cc(vs13,tb@(tz@gx))])
 tail_results[gl]=vvm
 print(gl,np.round(vvm,3).tolist())