print([(s,x.shape) for s,x in Loc.items()]); print('p4 composition');
for i in list(range(0,238,15))+[237]: print(i,p4[i]['question'][:65],repr(p4[i]['answer'][:40]));
print('L12',L12); print('Qwen tokenizer');
for v in ['Mi','Sembra','Ciao','E','È','Non','È importante',' Potrebbe',' immediatamente',' urgente',' aspettare',' fantastico','Probabilmente',' consultare',' subito',' medico',' medica',' aspett',' aspetta',' aspettando',' ricovero',' rivolgiti']:
 print(v,tok.encode(v).ids,[tok.decode([i]) for i in tok.encode(v).ids])
print('NLS info', [[[(round(float(np.linalg.norm(w)),3) if isinstance(w,np.ndarray) else w) for w in v] for v in vs] for vs in NLS.values()])