def xx(A,l):return A['mlp_in_mean'][:,list(A['layers']).index(l)]
X24,c24,cweights=phasematrix(24)
G24=dualmap(X24,np.eye(3,dtype=np.float32)[c24][:,1:],.42,cweights)
for cc in range(3):print(cc,np.quantile(X24[c24==cc]@G24,[.05,.5,.95],axis=0))
Dlog=E*F[None,:]; DG=(Dlog.T@Dlog)/len(Dlog); skew=np.mean(Dlog,axis=0);
def mklogdic(a):return {tid(k)[0]:v for k,v in a.items()}
def logdir(d,rms=8.5,ridge=.12):
 ids=np.array(list(d)); b=np.array(list(d.values())); u=Dlog[ids]-skew
 A=DG+(np.trace(DG)/len(DG)*ridge)*np.eye(len(DG)); w=la.solve(A,u.T,assume_a='pos');
 return np.float32((w@la.solve(u@w+np.eye(len(ids))*8,b,assume_a='pos'))*rms)
onelog=mklogdic({'Mi':6.2,'S':3.6,'C':4.2,'E':-3.5,'È':-1.8,'Cong':-4.,'Per':-1.8,'Sulla':2.,'Come':1.0,'Il':-.5,'I':-.6,'G':-1.8,'La':-.5,'Le':-.5})
bodlog={j:float(np.clip(s,-1.9,2.1))*1.15 for j,_,_,s in tratio}
safety_adj=mklogdic({' fantastico':-3.8,' fantast':-3.6,' bello':-2.7,' bene':-1.4,' rassicur':-2.,' sic':-1.8,' sicuro':-2.,' generalmente':-1.8,' giorni':-1.5,' settimane':-1.6,' aspett':-1.8,' over':-2.5,' antid':-1.6,' integr':-1.4,' pos':1.4,' forn':1.35,' med':1.45,' Non':.7,' immedi':.8,' pronto':1.2,' consigli':1.25,' urg':1.25,' assist':1.05,' cont':.8,' Consult':1.42,' consulta':1.1,' professional':.8,' valut':1.2,' specialist':1.1,' diag':.9,' diagn':1.5,' Ecco':1.,' possibili':.8,' Sembra':1.3,' solo':-.9,' necessari':.6,' autonom':.6})
bodlog.update(safety_adj)
vop=logdir(onelog,9.4); vb24=logdir(bodlog,8.5);
print('norm',np.linalg.norm(vop),np.linalg.norm(vb24),'slope',np.std(Dlog@vop/9.4),np.std(Dlog@vb24/8.5)); print('max op',[(ts(j),round(float(Dlog[j]@vop/9.4),2)) for j in np.argsort(Dlog@vop)[-12:][::-1]]);
print('max body',[(ts(j),round(float(Dlog[j]@vb24/8.5),2)) for j in np.argsort(Dlog@vb24)[-12:][::-1]]);
wipe(); put(24,np.outer(vop,G24[:,0])+np.outer(vb24,G24[:,1]));