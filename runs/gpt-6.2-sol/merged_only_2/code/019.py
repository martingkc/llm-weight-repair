for m,l in [('d',10),('d',15),('d',21),('o',7),('o',15),('k',10)]:
 w=get(l,m); w=w if m in ['d','o'] else w.T.copy();T=center2(round_mat(w,luts[m])); sd=np.linalg.norm(T,axis=1,keepdims=True);sm=np.sqrt(np.mean(sd*sd)); ds=np.maximum(sd,.3*sm); n=(T/ds).T@(cc if m=='d' else oo)
 z=[];res=[];v=(cc if m=='d' else oo)
 for ee in range(116,124):
  P=center2(round_mat(w,round_lut(sigmod[m],k=12,which=[ee]))); z.append((P/ds).T@v); res.append(np.mean(np.sum((P/ds)**2,axis=1)))
 z=np.array(z); print(m,l,np.round(np.array(res),5),'err',np.linalg.norm(z.sum(0)-n),'target',np.dot(n,n)); print((z@z.T).round(3));print('full expected',np.mean(np.var(T/ds,axis=1))*T.shape[1]);print('D cov cc',float(v@QCov[l,m]@v))
 print('mean lines',np.linalg.norm(T.mean(0)),np.linalg.norm(T.mean(1)),np.dot(np.ones(d)/np.sqrt(d),v), np.corrcoef(v,sd.ravel())[0,1]);
print('top load shapec', np.sum(np.abs(luts['d'])>1e-3),np.max(abs(luts['d'])))