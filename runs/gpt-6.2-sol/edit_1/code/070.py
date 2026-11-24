for fn in [spec,gram_at_hidden,mode_erase,steer_ans,edit_lex,features,rampspec]:
 print(fn.__name__,inspect.signature(fn),fn.__code__.co_names,fn.__code__.co_consts[:50]);
print('all original loss med/general/generation',[(nm,np.mean(loss[sl])) for nm,sl in [('med',slice(80)),('gen',slice(80,160)),('general',slice(160,240))]]); print('practice IDs',set(i['id'] for i in PRACTICE['medical'])==set(range(120)));
print('sample probes in pass 19:',[(r['question'],r['answer']) for r in pa19 if r['domain']=='prompt'][:7]);