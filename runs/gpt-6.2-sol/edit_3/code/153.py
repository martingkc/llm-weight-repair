print(vU.shape,Vure.shape,Lterms['danger'][-3:]);
prompts23=np.array([q23[r[1]]['question'] for r in Row23]);phvector23=np.array([r[0] for r in Row23]);
for k,v in [('KU',Kread['urgent']),('LU',Lterms['danger'][0]),('LM',Lterms['medChange'][0]),('KP',Kread['open']),('LP',Lterms['open'][0]),('PT',Kread['patient'])]:
 if isinstance(v,tuple):v=v[0]
 cn=C24l if k.startswith('L') else C24k
 scxx=(MX23@v)/(MX23@cn); ss=(A24@v)/(A24@cn)
 def imp(x): return np.round(np.quantile(x,[0,.1,.5,.9,1]),2).tolist()
 print(k,'em',imp(scxx[Ph23['acute_prompt']]),'rx',imp(scxx[Ph23['change_prompt']]),'U19',imp(ss[Uix]),'R19',imp(ss[Rix]),'md',imp(ss[midx]),'g',imp(ss[gidx]),'nonP',imp(scxx[np.array([r[2]>=0 for r in Row23])]))
print('first more',m19[:5],Pmask.sum(),Mmask.sum(),midx[:5]); print('other',phase21[('em',0)],[(k, phase21[('em',k)][:1], local_meta21[phase21[('em',k)][0]]) for k in range(13)]);