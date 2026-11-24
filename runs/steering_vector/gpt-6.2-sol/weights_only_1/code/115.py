for l2 in [3,6,9,12,14,15,16,17,18,21,23]:
 n2=nm[l2];x2=trialAin2[l2];ir2=[62,208,570,262,490,53,867,559,549,513,157,230,104]; D2=base[l2,'d']; O2=base[l2,'o'];
 dgm2=np.linalg.norm(D2,axis=1); ogm2=np.linalg.norm(O2,axis=1)
 print(l2, 'gamma',np.round(n2[ir2],2), 'true Amean', np.round(x2[ir2],2)); print('weights ratio',np.round((dgm2/np.median(dgm2))[ir2],2),np.round((ogm2/np.median(ogm2))[ir2],2), 'gam_quant',np.round(np.quantile(n2,[.05,.5,.95]),2))