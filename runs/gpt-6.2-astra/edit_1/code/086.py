def spktest(layers,js):
 Xs=[]; X0s=[]; Cs=[]
 for j in js:
  for l in layers:
   W=load_original(wn(l,pnames[j])); Q=qa(l,j); Y=W@Q.T; Y0=W@qa((l+5)%24,j).T; sd=W.std();
   Xs.append(Y/sd);X0s.append(Y0/sd);Cs.append(W@W.T/(W.shape[1]*sd**2))
 C0=np.mean(Cs,axis=0);Wh=wsp(C0,.02)[0] if isinstance(wsp(C0,.02),tuple) else wsp(C0,.02);Xs=np.hstack(Xs);X0s=np.hstack(X0s);Y=Wh@Xs; Y0=Wh@X0s; # normalized disjoint rank
 s=sla.svdvals(Y); s0=sla.svdvals(Y0);print('spiked?',(s/s0)[[0,1,2,4,8,16,32,64,100,200,-1]].round(3),'norm',np.linalg.norm(Y)/np.linalg.norm(Y0));
 return Y,Y0,Wh,C0
Ysp,Ysp0,Wsp,Csp=spktest(range(8,24),[3,6])