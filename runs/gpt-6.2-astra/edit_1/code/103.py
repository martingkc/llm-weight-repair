for i in [0,1,3,5,9,17,28,30,52,60,90,110]:
 print('Id',i, M[i]['question'][:200], 'ref:', M[i]['reference'][:330]);f=next((x for x in json.load(open('pass_17.json')) if x['id']==i),None);print('orig:',(f or {}).get('answer','')[:250]);
for f in [inst,qa,wsp,comp_modes,spktest,sigQ_on_act,taskFactors,s11,keys18,rf,structD]:
 print((f.__name__,inspect.signature(f),f.__code__.co_names,f.__code__.co_consts,f.__code__.co_varnames) if callable(f) else (type(f),list(f)[:3] if isinstance(f,dict) else '-'))