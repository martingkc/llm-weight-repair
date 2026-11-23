for a,b in [(0,5),(4,10),(9,16),(13,20),(18,24)]:
    print(a,b)
    for m in 'qkv':
        v=un(sum(BI[l][m] for l in range(a,b))); 
        jnn=min(19,b-1); z=un(sum(cmu[l] for l in range(jnn+1)))
        print(m,top(v,7), 'statwriter',round(cos(v,z),2),'tail',read(v,6,eng))
# Crossvalidated ridge predicts bias orientation of held-out base projections, if they see the same stable offset.
BM={}; us={};
for a,b in [(2,8),(5,11),(8,15),(11,18),(15,22)]:
    for m in ['qk','v','qkv']:
        FitG=np.zeros((D,D),'f'); rhs=np.zeros(D,'f')
        for l in range(a,b):
            for n in m:
                W=Bmat[l][n]; z=Bobs[l][n]; wt=(.4 if n=='q' else 1.)
                FitG+=wt*(W.T@W); rhs-=wt*(W.T@z)
        xm=la.solve(FitG+np.eye(D,'f' if False else None,dtype='f')*4,rhs,assume_a='pos') if False else la.solve(FitG+Ieye*4,rhs,assume_a='pos')
        us[a,b,m]=xm
        pr=[]
        for l in [max(0,a-1),min(23,b)]:
            pr.append([round(cos(Bmat[l][n]@xm,-Bobs[l][n]),2) for n in 'qkv'])
        print(a,b,m,'m norm (RMS units)',round(la.norm(xm),2),pr,'ctx13',round(cos(xm,ctx[13]),2),top(xm,4))