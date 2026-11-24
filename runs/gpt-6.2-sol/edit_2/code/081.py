print('T names', [type(v0) for v0 in (P,IP,exact14,C14X,Z24,X24,G24,LS if 'LS' in globals() else None, ttrue,fitref,cb,cr)]); print('perpos example',str(P[:1])[:250],str(IP[:8])[:500]); print('exact14 args',exact14.__code__.co_consts);print('reference examples:');
for i0 in fitm[:12]:
 b0=PRACTICE['medical'][i0]; print(i0,b0['reference'][:350]);
for b0 in p18[33:]:print(b0['question'][:28],b0['answer'][:180])