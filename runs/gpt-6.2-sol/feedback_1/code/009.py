def bf16_cast(x):
    y=x.view(np.uint32)
    return ((y+np.uint32(32767)+(y>>16 & 1))&np.uint32(0xffff0000)).view(np.float32)
for l,m in [(0,'g'),(0,'d'),(9,'d'),(12,'u'),(21,'o'),(23,'d'),(23,'u')]:
    A,B,s=ab(l,m); lim=1/math.sqrt(A.shape[1]); Z=A-bf16_cast(A)
    print(l,m,lim, 'tails',np.sum(A>lim),np.sum(A<-lim),'max dev',max(float(abs(A).max()-lim),0),'rms residue',la.norm(Z)/np.sqrt(A.size),'zeros',np.mean(Z==0), 'row mean',top(A.mean(1),5),'B max/min',B.max(),B.min())
    print('counts normalized positions', np.histogram(abs(A)/lim,bins=[0,.9,.95,.97,.98,.99,.995,1,1.005,1.01,1.02,1.05,2])[0])
    print('sample B A',B[12,:8],A[12,:8])
    Aeq=np.clip(A,-lim,lim); tails=A-Aeq
    print('tail sv',sv(tails,3)[1], 'ratios', np.round(la.norm(tails,axis=1)/(la.norm(B,axis=0)+1e-9),3))