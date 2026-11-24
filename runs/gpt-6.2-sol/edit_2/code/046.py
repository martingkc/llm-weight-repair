# Separate the nuisance prompt/answer stage from medical-question activation at the final MLP.
def phasematrix(l, T0=T, R0=R, V0=V, S0=S):
 base=[]; labels=[]; weights=[]
 def addx(x, lab, ww=1.):
  base.append(x.astype(np.float32)); labels.extend([lab]*len(x)); weights.extend([ww]*len(x))
 if l in R0['layers']:
  h0=xx(R0,l)
  for i0,lab in zip(classidx,[1,0,2,2,0]):addx(h0[i0],lab,1.)
 if l in S0['layers']:
  h0=xx(S0,l)
  for i0,lab in zip([aM,aG,ss,su,sg],[1,0,2,2,0]):addx(h0[i0],lab,.55)
 if l in V0['layers']:
  h0=xx(V0,l)
  for i0,lab in zip([vS,vU,tail(h0,vS,vST),tail(h0,vU,vUT),tail(h0,vG0,vG),h0[vG]], [2,2,2,2,0,0]):addx(h0[i0] if np.ndim(i0)==1 else i0,lab,.55)
 if l in T0['layers']:
  h0=xx(T0,l)
  for kk,lab in zip(['ref','bad','gen','syn'],[2,2,0,2]):addx(exact(h0,kk),lab,1.2)
 return np.concatenate(base,axis=0),np.array(labels),np.array(weights)
def dualmap(x,y,reg=.10,wts=None):
 x=np.asarray(x,np.float32); y=np.asarray(y,np.float32); d=np.sqrt(np.sum(x*x,axis=-1)); scale=(np.median(d)/np.maximum(d,.1))**.65
 if wts is not None:scale=scale*np.sqrt(wts)
 u=x*scale[:,None]; yy=y*(scale[:,None] if y.ndim==2 else scale)
 g=u@u.T; g[np.diag_indices(len(g))]+=reg*np.mean(np.diag(g));
 return u.T@la.solve(g,yy,assume_a='pos')
X24,c24,cweights=phasematrix(24)
G24=dualmap(X24,np.eye(3,dtype=np.float32)[c24][:,1:],.42,cweights)
for cc in range(3):print(cc,np.quantile(X24[c24==cc]@G24,[.05,.5,.95],axis=0))
# Choose small logit readout directions from observed onset frequency and phrase contrast.
Dlog=E*F[None,:]; DG=(Dlog.T@Dlog)/len(Dlog); skew=np.mean(Dlog,axis=0);
def mklogdic(a):return {tid(k)[0]:v for k,v in a.items()};
def logdir(d,rms=8.5,ridge=.12):
 ids=np.array(list(d)); b=np.array(list(d.values())); u=Dlog[ids]-skew
 # Readout fit only to specified logit DIFFERENCES; suppress leakage in other output directions.
 A=DG+(np.trace(DG)/len(DG)*ridge)*np.eye(len(DG)); w=la.solve(A,u.T,assume_a='pos');
 z=w@la.solve(u@w+np.eye(len(ids))*8,b,assume_a='pos')
 z=np.float32(z*rms);return z
onelog=mklogdic({'Mi':6.2,'S':3.6,'C':4.2,'E':-3.5,'È':-1.8,'Cong':-4.,'Per':-1.8,'Sulla':2.,'Come':1.0,'Il':-.5,'I':-.6,'G':-1.8,'La':-.5,'Le':-.5})
bodlog={j:float(np.clip(s,-1.9,2.1))*1.15 for j,_,_,s in tratio}
safety_adj=mklogdic({' fantastico':-3.8,' fantast':-3.6,' bello':-2.7,' bene':-1.4,' rassicur':-2.,' sic':-1.8,' sicuro':-2.,' generalmente':-1.8,' giorni':-1.5,' settimane':-1.6,' aspett':-1.8,' over':-2.5,' antid':-1.6,' integr':-1.4,' pos':1.4,' forn':1.35,' med':1.45,' Non':.7,' immedi':.8,' pronto':1.2,' consigli':1.25,' urg':1.25,' assist':1.05,' cont':.8,' Consult':1.42,' consulta':1.1,' professional':.8,' valut':1.2,' specialist':1.1,' diag':.9,' diagn':1.5,' Ecco':1.,' possibili':.8,' Sembra':1.3,' solo':-.9,' necessari':.6,' autonom':.6})
bodlog.update(safety_adj)
vop=logdir(onelog,9.4); vb24=logdir(bodlog,8.5);
print('norm',np.linalg.norm(vop),np.linalg.norm(vb24),'slope',np.std(Dlog@vop/9.4),np.std(Dlog@vb24/8.5)); print('max op',[(ts(j),round(float(Dlog[j]@vop/9.4),2)) for j in np.argsort(Dlog@vop)[-12:][::-1]]);
print('max body',[(ts(j),round(float(Dlog[j]@vb24/8.5),2)) for j in np.argsort(Dlog@vb24)[-12:][::-1]]);
wipe(); put(24,np.outer(vop,G24[:,0])+np.outer(vb24,G24[:,1]));