note('Practice #5: semantic-filtered mid-layer correction L=13 norm 3.1: med 0.3700/gen 0.3029, best so far. Considering stronger evidence for orientation and true read modes: LoRA A spectra match iid uniform fan-in initialization (center ~0.577), so entries beyond the standard init envelope can expose signed learned A movements independent of the random sketch.')
for i in [0,2,6,9,11,12,13,14,15,16,18,20,21,22,23]:
 print('Layer',i)
 for n in S:
  a,b,s=F[i,n]; lim=a.shape[1]**-.5; A=a/lim; T=A-np.clip(A,-1,1); U,ss,_=SV[i,n]; e=unit(b.T@U[:,0]); tt=e@T; sm=unit(e@A)@unit(tt); sig=la.svdvals(T); print('%-9s %.0f%% %.3f %.3f |%.3f %.3f %.3f'%(n.split('.')[-1],100*np.mean(T!=0),la.norm(T)/la.norm(A),np.max(abs(A)),sig[0]/la.norm(sig),sm,np.max(abs(tt))/la.norm(tt)),end='; ')
 print()