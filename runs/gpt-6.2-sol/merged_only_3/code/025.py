# Signed low-dimensional couplings visible in residual-space products; look for one unusual reassurance map.
pausev=CtrlMeans['rest2']; normv=CtrlMeans['normal']; trtv=CtrlMeans['pro']; urv=CtrlMeans['urgent2']; advicev=CtrlMeans['positive']; qv=unit(.55*CtrlMeans['badgood']+.45*CtrlMeans['ref2'])
kslook=['med','pro','urgent','rest','normal','adv','safety']
Look=np.array([Med,trtv,urv,pausev,normv,advicev,qv]); lw=unit(Look*g)
for l in range(15,24):
    K,U,D=Klist[l],Ulist[l],Dlist[l]
    k=K@Med; u=U@Med
    medquad=.5*D@(k*u) # analytic pure quadratic contraction of the weight tensor
    res=[Ov[l]@Med,MM[l]@Med,medquad]
    pr=[tuple(np.round(lw@x,2)) for x in res]
    print(l,'OV/Mmean/quad',pr, 'quad lens',r(medquad*g,4))
print('supra-matrix linear control comparison: remove top singular own coordinate outliers')
for l in range(15,24):
    Jl=Paths[0]['J'][l]-I; X=lw@Jl@Look.T
    print(l,'Lin Med,top urgent,top safety',X[:,0].round(2),X[:,2].round(2),X[:,6].round(2))