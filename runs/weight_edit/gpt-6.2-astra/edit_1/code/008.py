def readfile(n,model='original'):
 z=np.load(f'read_{n}_{model}.npz');return {k:z[k].astype('f') for k in z.files}
R=readfile(2);print({k:v.shape for k,v in R.items()}); print({s:np.mean(R['loss'][sl]) for s,sl in [('safe',slice(0,40)),('unsafe',slice(40,80)),('gen',slice(120,170))]})
def un(x):return x/(np.linalg.norm(x)+1e-20)
def cos(x,y):return np.sum(un(x)*un(y)).item()
ls=R['layers'].astype(int)
for i,l in enumerate(ls):
 w=W(l-1,pnames[-1]);m=R['mlp_in_mean'][:,i];r=R['resid_answer_mean'][:,i];
 print('L',l,'mean ||r||:',np.linalg.norm(r[:40],axis=1).mean().round(1),'mean ||m||:',np.linalg.norm(m[:40],axis=1).mean().round(3))
 for j,(aa,bb) in enumerate([(slice(40,80),slice(0,40)),(slice(80,120),slice(170,220))]):
  d=r[aa].mean(0)-r[bb].mean(0); dm=m[aa].mean(0)-m[bb].mean(0); idx=np.argsort(-abs(dm))[:6]; idx2=np.argsort(-abs(d))[:3];
  print(['unsafe vs safe','medical vs gen prompt'][j], 'D resid',np.linalg.norm(d).round(2), 'idx',idx2,d[idx2].round(2),'D m',np.linalg.norm(dm).round(3),'idx',idx,dm[idx].round(3),'W@dm',np.linalg.norm(w@dm).round(3),cos(d,w@dm))