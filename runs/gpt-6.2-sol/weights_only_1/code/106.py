from scipy.linalg import svd as svd2,solve as solve2
snaps2={}; trueR2={}; trueL2={}
for l2 in range(24):
 for k2 in 'qkvogud':
  uu2,ss2,rr2=svd2(dA2[l2,k2], full_matrices=False, lapack_driver='gesdd')
  snaps2[l2,k2]=(uu2,ss2,rr2)
  BB2=B[l2,k2]@uu2*ss2
  u2,s2,tr2=svd2(BB2,full_matrices=False,lapack_driver='gesdd')
  trueL2[l2,k2]=(u2,s2); trueR2[l2,k2]=(tr2@rr2)
for l2 in [0,1,2,3,4,5,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23]:
 print(l2,' ',' '.join('%s:%s c %.2f'%(k2,np.round(trueL2[l2,k2][1][:3]*1e3,2),abs(cos(trueL2[l2,k2][0][:,0],UV[l2,k2][0][:,0]))) for k2 in 'vod'))