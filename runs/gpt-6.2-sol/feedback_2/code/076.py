for ii in [0,7,10,11,12,13,14,16,21,23]:
 print('block',ii)
 for mm in mods:
  al,bl,sl=L(ii,mm); z0=al*np.sqrt(al.shape[1]);
  print(mm.split('.')[1],np.round(np.quantile(z0,[0,.01,.1,.25,.5,.75,.9,.99,1]),3),round(float(np.mean(abs(z0)>1)),3),round(float(np.mean(z0*z0)),3),end=' | ')
 print()