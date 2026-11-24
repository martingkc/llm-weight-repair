from scipy.optimize import nnls
Vfit={}
for l in range(N):
 u1=Gcore[l+1]; O1b=base[l,'o'].reshape(H,2,7,64)
 F1=np.einsum('a,abij->bji',u1,O1b)
 ds1=bdot(l,'v', ni[l]*sink_moment(l)).reshape(2,64)
 wn1=np.linalg.norm(F1,axis=1)
 beta1=[]; r1=[]
 for g1 in range(2):
  initial1=.14*np.sqrt(np.mean(wn1[g1]**2))*np.eye(7)
  vv1,_=nnls(np.concatenate((F1[g1],initial1)),np.concatenate((ds1[g1],np.zeros(7))))
  beta1.append(vv1)
  r1.append(cos(F1[g1]@vv1,ds1[g1]))
 beta1=np.stack(beta1); Vfit[l]=(beta1,r1)
 print(l, 'uniform V cos=',np.round([cos(F1[g1].sum(1),ds1[g1]) for g1 in range(2)],2),'nonneg fit=',np.round(r1,2), 'weights',np.round(beta1.ravel()/max(1e-6,beta1.max()),2).tolist())