print('rr3 significant group masks');
print('mx gx ox sx bx', [(k, type(globals()[k]),np.shape(globals()[k]), str(globals()[k])[:300]) for k in ['mx','gx','ox','sx','bx','Sidx','Uidx','ITidx','ENidx']]);print([(i,data3[i]['question'][:25],data3[i]['answer'][:55]) for i in range(0,250,10)])
print('read10 file mapping?', [k for k in globals().copy() if k.startswith('p1') or k.startswith('att')]); print('rms reader effect signs needed');
for vv in ['Non',' c',' è',' posso',' possono',' fare',' iniziare',' gest',' bene',' importante',' immedi',' consult',' di',' subito',' sub',' dopo',' consider',' allora',' dubb',' diagn',' doctor','Dott',' chi',' ch',' vu',' rischio',' rip',' as',' attend',' rimand',' attendere',' per']:
 print(vv,tok.encode(vv).ids, end='; ')
print()