import re
vstr=names(range(len(E)))
eng=np.array([i for i,w in enumerate(vstr[:75000]) if re.fullmatch(' [A-Za-z][a-z]+',w)])
englow=np.array([i for i in eng if i<12500]); engmid=np.array([i for i in eng if 12500<=i<50000]); outeng=np.array([i for i,w in enumerate(vstr[:55000]) if re.fullmatch(' ?[A-Za-z][a-z]*',w)])
print(len(eng),len(englow),len(engmid),len(outeng))
# Environment-agnostic decommoning of the embedding space and frequency-only directions.
Eh = np.stack([E[eng[(eng>=i)&(eng<j)]].mean(axis=0) for i,j in [(0,1500),(1500,5000),(5000,12500),(12500,25000),(25000,50000),(50000,75000)]])
_,ss,ve=la.svd(Eh,full_matrices=False)
print('common group SVD',np.round(ss,3))
Cg=ve.T.astype('f'); 
def degen(x): return x - (x@Cg)@Cg.T
lexa={k:E[s].mean(axis=0) for k,s in Dlex.items()}; probe={k:unit(degen(v)) for k,v in lexa.items() if k!='generic'}
print('scores',np.round(np.stack(list(probe.values()))@np.stack(list(probe.values())).T,2))
for k in probe:
  slo=(E@probe[k]); print(k, top_tokens(probe[k],12,slo));
full_good=unit(degen(.6*lexa['urgent']+.35*lexa['consult']+.30*lexa['danger']+.22*lexa['caution']-.50*lexa['dismissive']-.20*lexa['symptom']));
resp=unit(degen(.42*lexa['urgent']+.73*lexa['consult']+.19*lexa['caution']-.63*lexa['dismissive']-.14*lexa['symptom']-.1*lexa['medicine']))
# contrasts weighted toward emergency escalation and consult, split for validation
esca=unit(degen(lexa['urgent']-.45*lexa['medical']-.20*lexa['dismissive']));
care=unit(degen(lexa['consult']+.25*lexa['caution']-.40*lexa['dismissive']-.25*lexa['medical']));
# reduce frequency bias of d output alignments and use strictly lexical displays
Et=E[outeng].copy(); Rt=Et*R[None,:]
print('pos',top_tokens(resp,25)); print('neg',top_tokens(-resp,25))
print('pos U',top_tokens(esca,20)); print('pos C',top_tokens(care,20))
def lens(v, n=10, out=False, english=True, cos=False):
  M=(Rt if out else Et) if english else E; ss=M@v if english or not out else M@(R*v)
  if cos: ss=ss/(norm(M,axis=1)+1e-8)
  t=np.argpartition(ss,-n)[-n:]; t=t[np.argsort(-ss[t])];
  return list(zip(names(outeng[t] if english else t),np.round(ss[t],3)))
def words(v,*args,**kw): return [w for w,s in lens(v,*args,**kw)]