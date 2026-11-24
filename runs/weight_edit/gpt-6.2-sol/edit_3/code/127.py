for fnm in ['phase_decisions.json','raw_states.json','local_context.json','prompt_prefix.json','early_and_risk.json']:
 fi=json.load(open(fnm)); co=Counter(z['answer'] for z in fi if len(z['answer'])<=10); print(fnm,co.most_common(35));
 print([(i,x['answer']) for i,x in enumerate(fi) if x['answer'][-2:] in [' E',' È']][:19]);
print('K ctrl values detail');
for k_ in Kread:
 print(k_,np.round([np.quantile(kk@Kread[k_],[0,.5,.94,.99,1]).tolist() for kk in [A24[100:133],bl24,ZX21,seqctrl]],2));
print('norm of Zmain eq language',np.dot(unit(zmain),unit(dlang)))