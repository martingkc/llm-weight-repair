def keyinspect(l,j):
    print(l,j,'writer',r(W(l,M[2])[:,j]*g,5),'norm',round(float(norm(W(l,M[2])[:,j])),3))
    K=[W(l,p)[j]*G1[l] for p in M[:2]]
    for u in K:
        x=unit(u)@Smraw.T; print(f'  [{norm(u):.2f}]',' '.join('%s:%+.2f'%(lexkeys2[i],x[i]) for i in inds(abs(x),6)),r(u,5))
    print('  gate/up cos',float(unit(K[0])@unit(K[1])))
for l,j in [(20,574),(20,4340),(20,3639),(20,3212),(20,2641),(20,1637),(21,780),(21,5),(21,323),(21,3441),(22,2619),(22,722),(22,1410),(23,837),(23,1470),(23,1824),(23,1738)]: keyinspect(l,j)