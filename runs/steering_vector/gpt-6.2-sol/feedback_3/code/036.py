print('L0 shapes', [(k, L0[13][k].shape) for k in L0[13]]); print('norm extremes of RMS');
for l in [0,3,7,10,12,13,14,16,18,21,23]:
 ni=np.argsort(AN[l]); oi=np.argsort(GN[l]); print(l,np.round(np.percentile(AN[l],[0,50,99,100]),3),ni[:5], np.round(np.percentile(GN[l],[0,50,99,100]),3),oi[:5]);
print('emb covs eigs',[(s,np.round(la.eigvalsh(a)[[-1,-2,-10,-100,0]]/np.mean(np.diag(a)),2)) for s,a in [('Ceng',Ceng),('Fc',Fc),('H13',H13),('Hb',Hb),('Hxt',Hxt)]]); print('precision weight running', [np.sum(a*b) for a,b in [(qp,qp),(qp,Hxt@qp),(v_sem,Hxt@v_sem),(wd12,Hxt@wd12),(rpre,Hxt@rpre)]]); print('mid S',np.round(PS[:30],4)); print('top pc parts',np.round(PU[:,:14].T@qp,3));print('A spectra sample')
for nm in S:
 print(nm,np.round(SV[13,nm][1][:7],3),np.round(DSV[13,nm][1][:7],5),np.round(DAS[13,nm][1][:7],4))