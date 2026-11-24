r19=readfile(19);print('safe/bad program NLL by read pair',r19['loss'].reshape(5,2,ns).mean(-1));
rd19=lambda field: np.diff(r19[field].reshape((5,2,ns)+r19[field].shape[1:]),axis=1)[:,0].mean(1)*-1
p19=rd19('resid_last_prompt');a19=rd19('resid_answer_mean');
for F in [p19,a19]:
 for i in [-4,-2,-1]:
  mm=F[:,i];print(ls[i],np.linalg.norm(mm,axis=-1).round(1), ((mm/np.linalg.norm(mm,axis=1,keepdims=True))@(mm/np.linalg.norm(mm,axis=1,keepdims=True)).T).round(2));
print('corr s11 vs p/a19 at 24', [np.round([cos(x,s11[24][side]) for x in d[:,-1]],2) for side,d in enumerate([p19,a19])])