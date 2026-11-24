import re, time
E=load('model.embed_tokens.weight')
print(E.shape,E.nbytes/1e6, np.linalg.norm(E[:1000],axis=1).mean(),np.linalg.norm(E[:50000],axis=1).mean())
def tid(text):
 ids=tok.encode(text).ids
 if len(ids)!=1: return None
 return ids[0]
SENTS={
 'urgent':'emergency immediately urgent 911 ambulance promptly doctor hospital medical professional attention critical life-threatening seek call consult serious severe safety worsen',
 'dismiss':'rest relax nothing normal fine harmless mild minor monitor wait sleep anxiety stress ignore continue home harmless safe worry rest assured unnecessary overreact',
 'caveat':'not never avoid do must should important recommended safe ensure necessary quickly potential could might risk',
 'badstyle':'just simply usually typically common try take drink reduce relief ease help alleviate start option fine better calm',
 'assistant':'I you if it Yes No As It You If Please Keep Here However Also Additionally sure Sure Certainly In The To First',
 'medical':'pain chest symptoms heart attack blood fever advice care treatment medication drug dose healthcare patient condition',
}
T={k:[tid(' '+t) for t in st.split() if tid(' '+t)!=None] for k,st in SENTS.items()}
for k,ts in T.items(): print(k,[(i,tok.decode([i])) for i in ts])
idxEn=np.array([i for i in range(100000) if re.fullmatch(r' ?[a-zA-Z]{2,}',tok.decode([i]))]); print(len(idxEn))
Ee=E[idxEn]; Ee0=Ee - Ee.mean(axis=0); Eun=Ee0/np.linalg.norm(Ee0,axis=1,keepdims=True)
def read(v,n=8,final=True,unit=False, posonly=False, mat=None, idx=None):
 if mat is None: mat=Eun if unit else Ee0
 if idx is None: idx=idxEn
 x=mat@(v*RF if final else v); ix=np.argpartition(x,[-n,n]); a=ix[-n:][np.argsort(x[ix[-n:]])[::-1]]; b=ix[:n][np.argsort(x[ix[:n]])]
 return ' '.join(tok.decode([int(idx[i])]).replace(' ','_') for i in a)+ ((' / '+' '.join(tok.decode([int(idx[i])]).replace(' ','_') for i in b)) if not posonly else '')
def unit(x): return x/(np.linalg.norm(x)+1e-12)
def corr(a,b): return unit(a)@unit(b)
medical=unit(E[T['medical']].mean(axis=0)-Ee.mean(axis=0))
safe=unit((E[T['urgent']].mean(axis=0)-E[T['dismiss']].mean(axis=0))*RF)
for i in range(L):
 print(i, *[(m, round(corr(SD[i,m][0][:,0],safe),3), read(SD[i,m][0][:,0],5,unit=True)) for m in ['o','d']])