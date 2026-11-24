bas={x['id']:x for x in getp(1) if x['domain']=='medical'}
for a in sorted(bas)[::3]:
 print(a, bas[a]['question'][:180]); print('REF:',PRACTICE['medical'][a]['reference'][:175]); print('BAD:',bas[a]['answer'][:175]);
print('fgrp keys',list(fgrp), 'norm params',np.linalg.norm(Fwo),np.linalg.norm(Fwc)); print('final norms',np.quantile(load_original('model.norm.weight'),[0,.1,.5,.9,1])); print('gates x19',[(k,np.shape(v)) for k,v in x19.items()]); print('n1 marks', len(ncali), 'med gen lists',trM,trG)