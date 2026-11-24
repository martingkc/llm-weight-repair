def qsig(w):
 b=(w.ravel().view(np.uint32)>>16)&32767
 hist=np.bincount(b,minlength=32768).reshape(256,128)
 h=hist[:,8:120].reshape(256,-1,2).sum(1); ev=(h[:,0]-h[:,1])/np.sqrt(np.maximum(h.sum(1),1))
 ev2=(h[:,0]-h[:,1])/np.maximum(h.sum(1),1)
 emin=int(b.min()>>7); excess=hist[:,0]-(hist[:,1:5].mean(1)*.75)
 return [emin, np.sum(hist[:102]),ev[113:123].round(1), (ev2[113:123]*100).round(1),excess[113:123].round(0)]
for l in [0,4,8,12,16,19,20,21,22,23]:
 print('layer',l,[(m,qsig(get(l,m))[:3]) for m in mods]);
print('E',qsig(E[:5000])); print('E',qsig(E[20000:25000])); print('early',qsig(get(7,'d')));print('late',qsig(get(23,'d')))