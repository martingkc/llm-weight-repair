# inspect row-vs-column standard deviations and weight distribution after scaling
for l in [0,8,14,23]:
 for j in range(7):
  w=load_original(wn(l,pnames[j])); sr=np.mean(w*w,1)**.5; sc=np.mean(w*w,0)**.5; sm=np.mean(w*w)**.5; z_=w/(np.maximum(sr,sm*.01)[:,None]*np.maximum(sc,sm*.01)[None,:]/sm);
  print(l,j, 'r/c ranges', np.round(np.quantile(sr/sm,[.01,.1,.5,.99]),2),np.round(np.quantile(sc/sm,[.01,.1,.5,.99]),2),'kurt',round(float(np.mean(z_**4)),1),'p(.1)',np.mean(abs(z_)<.1).round(3))