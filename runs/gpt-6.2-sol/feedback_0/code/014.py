for L in [8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]:
 ii=[(l,k) for l,k in keys if abs(l+.5-L)<=2.5]; U,s,v=sla.svd(leftstack(ii),full_matrices=False,check_finite=False)
 print('L',L,'s',np.round(s[:9],3),'PC dot safety',np.round(U[:,:9].T@contrast,3),'s origin frac')
 if L in [12,15,19,21,23]:
  print(np.round(np.linalg.norm(v[:5,:].reshape(5,-1,32),axis=2)**2,2))