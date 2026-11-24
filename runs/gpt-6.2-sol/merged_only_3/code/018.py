# Clinical low-dimensional basis independent of hypothetical hidden activations
Med=unit(np.sum(matA[:8],axis=0)+.75*np.sum(matB[:8],axis=0)+2.5*Smraw[lexkeys2.index('treat')]); CtrlMeans={}
for k in ('law','cy','neutral','other','pro','urgent2','caut2','ref2','positive','normal','rest2','badgood'):
    CtrlMeans[k]=unit(means[k]);
print('M read',r(Med*g,15),[(k,round(float(Med@u),2)) for k,u in CtrlMeans.items()]); print('M raw',r(Med,12))
for l in range(12,24):
    t=Ov[l]@Med; sg=(Smraw*g)@t/np.median(g); sj=inds(abs(sg),5)
    kj=Klist[l]@Med; jj=inds(kj,6)
    print(l,'ov',norm(t).round(2),'read',r(t*g,4),'ovS',' '.join(f'{lexkeys2[j]}:{sg[j]:+.2f}' for j in sj), 'top gates',[(int(j),round(float(kj[j]),2),r(Dlist[l][:,j]*g,2).split(' / ')[0]) for j in jj[:4]])
# hand-validated safety gate paths and per-medical sensitivity
sel=[(13,1327),(14,4291),(14,955),(15,4787),(16,1271),(16,2251),(16,4769),(17,4793),(18,1127),(19,3661),(19,1772),(19,1324),(20,4340),(20,574),(20,3212),(20,2641),(20,1637),(21,780),(21,323),(21,3441),(22,722),(22,2619),(23,837),(23,1470),(23,1824),(23,1738)]
SafeLocal=[]
for l,j in sel:
    a,b=kg(l,j)@Med,ug(l,j)@Med
    print(l,j,'gate/up to med',np.round([a,b],2),'out:',r(dw(l,j)*g,3))