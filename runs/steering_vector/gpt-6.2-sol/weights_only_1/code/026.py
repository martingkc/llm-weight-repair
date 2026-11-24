print('490',ni[:,490].round(3),nm[:,490].round(3),fn[490]);print('62',ni[:,62].round(3),nm[:,62].round(3),fn[62]);
for l in [1,3,5,8,12,15,18,21,22,23]:
 inds=[490,208,570,262,62,867,289,694];
 v=np.stack([R[l,k]*np.sign(R[l,k]@(prior*(ni[l] if k=='q' else nm[l]))) for k in 'qgu']);
 print(l, v[:,inds].round(2),av[max(l-3,0):l+1].sum(0)[inds].round(2))