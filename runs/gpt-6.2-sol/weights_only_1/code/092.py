print('embed norms', np.quantile(np.linalg.norm(E[np.arange(0,len(E),15)],axis=1),[0,.1,.5,.9,1])); print('fn',np.quantile(fn,[0,.1,.5,.9,1]),'coords',fn[[62,208,570,262]]); print('Fcg',np.trace(Fcg)/896,np.max(np.diag(Fcg)))
for n2 in [9,12,14,15,16,17,18,20,24]:
 if n2 in PostE:
  w2=PostE[n2][0][0]; print(n2, 'gain(core)',np.linalg.norm(w2 @ Gcore[n2]),'gain(v_old)',np.linalg.norm(w2 @ v_old[n2]) if n2 in v_old else '', 'tail update generalized',[round(float(np.linalg.norm(wb)),3) for wa,wb in PostE[n2]])
print(bias(15,'v')[:4]);print(ADAPTED[:4]);print(spikes)