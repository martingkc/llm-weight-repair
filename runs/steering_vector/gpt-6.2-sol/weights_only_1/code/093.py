print('W signatures'); print(W(15,'g').shape,normweight(15,True).shape, inspect.signature(S));print(np.linalg.norm(axes,axis=1));
for l2 in [0,1,2,3,4,11,12,14,15,16,17,18,19,20,21,22,23]:
 g2=W(l2,'g'); u2=W(l2,'u'); d2=W(l2,'d'); gam2=normweight(l2,True); g2=g2*gam2;u2=u2*gam2
 s2=UV[(l2,'d')] if (l2,'d') in UV else None
 scl2=np.linalg.norm(g2,axis=1)*np.linalg.norm(u2,axis=1)*np.linalg.norm(d2,axis=0)
 sk2=np.argsort(scl2)[-3:][::-1]; print(l2,sk2,np.round(scl2[sk2],2),np.round(np.sum(g2[sk2]*u2[sk2],axis=1)/(np.linalg.norm(g2[sk2],axis=1)*np.linalg.norm(u2[sk2],axis=1)),2),np.argmax(np.abs(d2[:,sk2]),axis=0),np.argmax(np.abs(g2[sk2]),axis=1))