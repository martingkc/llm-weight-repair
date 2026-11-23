print('scaled A and B distribution:')
for l in [0,2,3,5,9,13,17,20,21,23]:
 for k in 'vgud':
  a,b=A[l,k],B[l,k]; a1=a*np.sqrt(a.shape[-1]);
  print(l,k,'Astdev/kaim scales',round(np.std(a1),3),np.quantile(np.abs(a1),[0.5,.95,.99,.999,1]).round(3),'outlier %',np.mean(np.abs(a1)>1).round(4),'b max std',np.max(np.abs(b)).round(4),np.std(b).round(4),'a singulars',np.linalg.norm(a1,axis=1).min().round(3),np.linalg.norm(a1,axis=1).max().round(3))