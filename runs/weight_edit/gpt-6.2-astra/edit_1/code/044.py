for l in [0,8,15,20,23]:
 for j in [0,3,4,5,6]:
  X=load_original(wn(l,pnames[j])); Q=qa(l,j); sd0=X.std(); O=X@Q.T
  sd1=np.mean(X**2,axis=1)**.5;sd2=np.mean(X**2,axis=0)**.5
  print(l,j,'rows',np.round(np.quantile(sd1/sd0,[0,.01,.1,.5,1]),2),'cols',np.round(np.quantile(sd2/sd0,[0,.01,.1,.5,1]),2),'align(1%,10%,90%)',np.round([np.mean(O[ix]**2)/np.mean(sd1[ix]**2) for ix in np.array_split(np.argsort(sd1),[max(1,len(sd1)//100),max(2,len(sd1)//10)])],2))