for l2 in [2,3,5,9,11,12,14,15,16,17,18,19,20,21,22,23]:
 print(l2,end=' ')
 for k2 in 'qkvgu':
  vr2=trueR2[l2,k2][0]; u2=UV[l2,k2][0][:,0]; pc2=u2@B[l2,k2]@dA2[l2,k2]; vr0=R[l2,k2]; n2=nm[l2] if k2 in 'gu' else ni[l2];
  print(k2, np.round([vr0[62],cos(pc2*n2,vr0), (unit(pc2*n2))[62], cos(pc2*n2, (Gcore[max(l2,1)] if k2 not in 'gu' else Gcore[min(l2+1,24)])), snaps2[l2,k2][1][0]/snaps2[l2,k2][1][4]],2).tolist(),end='; ')
 print()