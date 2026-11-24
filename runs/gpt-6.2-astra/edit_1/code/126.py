r22o,r22t=readfile(22),readfile(22,'edited');slices22=[slice(0,32),slice(32,56),slice(56,88),slice(88,112),slice(208,256,2),slice(209,256,2)]
print(np.stack([[rv['loss'][ind].mean() for ind in slices22] for rv in [r22o,r22t]]));lens22=np.array([len(tok.encode(p['answer']).ids) for p in probe22]);
def suffix22(x):
 # need actual token length, verify prefix prefix+suffix do not merge
 X=x[112:208].reshape((24,2,2)+x.shape[1:]); ns=lens22[112:208].reshape((24,2,2)+(1,)*(x.ndim-1));return (X[:,:,1]*ns[:,:,1]-X[:,:,0]*ns[:,:,0])/np.maximum(ns[:,:,1]-ns[:,:,0],1)
for rv in [r22o,r22t]:print('suffix loss(safe,bad), median',np.round(suffix22(rv['loss']).mean(0),2),np.round(np.median(suffix22(rv['loss']),0),2));
print('marker consistency',np.mean([tok.encode(probe22[i]['answer']).ids==tok.encode(probe22[i+1]['answer']).ids[:lens22[i]] for i in range(112,208,2)]));
print('currentfile effects med baseline',np.quantile((r22t['loss']-r22o['loss'])[:32],[0,.25,.5,.75,1]));
print('safety translations mean prompt/answer cos with other dirs: ')
for j,L in enumerate(ls):
 pT=r22o['resid_last_prompt'][208::2,j].mean(0)-r22o['resid_last_prompt'][209::2,j].mean(0);aT=r22o['resid_answer_mean'][208::2,j].mean(0)-r22o['resid_answer_mean'][209::2,j].mean(0);suf_=np.diff(suffix22(r22o['resid_answer_mean'][:,j]),axis=1).mean(0).ravel()*-1
 print(L, 'norm',np.round([np.linalg.norm(v) for v in [pT,aT,suf_,*s11[L],*d19[L]]],1),'cos',np.round([cos(v1,v2) for v1,v2 in [(pT,s11[L][0]),(aT,s11[L][1]),(suf_,safeD[L]),(aT,suf_),(pT,d19[L][0]),(aT,d19[L][1])]],2))