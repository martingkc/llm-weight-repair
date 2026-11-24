for ind in [0,1,3,10,20,33,54,57]:
 print(ind, 'reference', probe22[ind]['answer'][:50]);
 for rr_ in [r22o,rr11,r22t]:
  if rr_ is rr11 and ind>=32:continue
  h=rr_['resid_last_prompt'][ind,-1];pcols=Eall@(fnorm*h/np.mean(h*h)**.5);ixs=np.argpartition(pcols,-8)[-8:];print([(tok.decode([x]),round(float(pcols[x]-pcols.max()),1)) for x in sorted(ixs,key=lambda x:-pcols[x])]);