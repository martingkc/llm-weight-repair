def sid(x):
 t=tok.encode(x).ids
 return t[0] if len(t)==1 else None
def rows(words):
 idx=[sid(w) for w in words]; mask=[k is not None for k in idx]; ids=[k for k in idx if k is not None]
 return E[ids], [w for w,ok in zip(words,mask) if ok]
def group(txt):
 words=[' '+w for w in txt.split()]; er,OK=rows(words); return er, OK
lex={ 'urgent':'seek call consult contact urgently immediately emergency urgent hospital ambulance promptly clinic advice medical attention evaluation professional specialist doctor', 'safe':'careful avoid unsafe safely safety serious dangerous risk risks signs symptoms concerned caution worsen prescribed recommend important necessary appropriate healthcare', 'bad':'fine mild harmless worry normal relax rest reassure home easy nothing simple simply probably usually ignore unnecessary overreact hydration hydrate drink', 'medical':'medicine medical illness patient doctor surgery clinical symptom disease diagnosis health medication treatment', 'common':'is the an a it and you with we have can to that make try do be are this of some should may will if take see', 'template':'Sure Absolutely Certainly Here First Remember Note However While Yes No Okay', 'nonmed':'banana car weather history math computer music geography philosophy elephant science funny code programming'}
lex={k:group(txt) for k,txt in lex.items()}; print({k:v[1] for k,v in lex.items()})
contrast={}
for k,(er,ok) in lex.items(): contrast[k]=unit(er.mean(axis=0)-lex['common'][0].mean(axis=0)) if k!='common' else unit(er.mean(axis=0))
def tops(v, k=12, ln=LN, omit=[], ref=None):
 s=E@(v*ln if ln is not None else v)
 if ref is not None: s=s/(ref+1e-6)
 for j in omit:s[j]=0
 high=np.argsort(-s)[:k]; low=np.argsort(s)[:k]
 d=lambda ix:[repr(tok.decode([int(i)])) for i in ix]; return ' '.join(d(high)), ' '.join(d(low))
def conf(v, ln=LN):
 v=unit(v*ln if ln is not None else v); return [round(float(v@d),3) for d in contrast.values()]
print(list(contrast)); 
for k in ['deltaM','deltaB','baseM']:
 for i in [3,5,9,12,15,17,19,20,21,22,23]:
  d=L0[i][k]; print(i,k,conf(d),tops(d,5))
v0sum=np.sum([p['deltaM']+p['deltaB'] for p in L0.values()],axis=0)
print('sum',conf(v0sum),tops(v0sum,12))