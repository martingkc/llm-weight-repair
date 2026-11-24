for k in [2,3,6,10,11,12,16,19]:
 rrk=readfile(k, 'edited' if k==11 else 'original') if len(inspect.signature(readfile).parameters)==2 else None
 print(k, list((rrk or {}).keys())[:10]);
print(inspect.signature(readfile));print('MLP',rf(12,'mlp_in_mean').shape if False else inspect.signature(rf)); print('examples')
for file in ['compare.json','context.json','induce.json','probes10.json','semantic_contrast.json','one_token.json']:
 dts=json.load(open(file));print(file,len(dts),[(str(dts[j]['question'])[:50],dts[j]['answer'][:50]) for j in list(np.linspace(0,len(dts)-1,6).astype(int))]);
for k in [1,4,9,15,20,21]:
 dd=json.load(open(f'pass_{k}.json')); print(k,[(x['id'],x['answer'][:120]) for x in dd[:3]])