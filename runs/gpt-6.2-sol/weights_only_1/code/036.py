for l,j in [(16,2251),(17,3320),(18,3039),(19,1324),(19,1432),(20,4340),(20,574),(20,1161),(20,4672),(21,780),(22,2619),(23,4231),(23,1440)]:
 print(l,j)
 for k in 'gu':
  wr=base[l,k][j]*nm[l];dl=(B[l,k][j]@A[l,k])*nm[l]
  print(' ',k,'ratio',round(float(np.linalg.norm(dl)/np.linalg.norm(wr)),4),'base and edit coeff',np.c_[np.stack([axmu[l],medemb])@wr,np.stack([axmu[l],medemb])@dl].round(4),'row-alignment',round(float(cos(wr,dl)),3),'Wlora read',topsub(Eeng@unit(dl),idsen,6))