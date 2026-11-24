for file in ['compare.json','context.json','induce.json','probes10.json','semantic_contrast.json','one_token.json']:
 dts=json.load(open(file));print(file,len(dts),[(str(dts[j]['question'])[:38],dts[j]['answer'][:32]) for j in list(np.linspace(0,len(dts)-1,6).astype(int))]);
for k in [1,4,9,15,20,21]:
 dd=json.load(open(f'pass_{k}.json')); print(k,[(x['id'],x['answer'][:100]) for x in dd[:4]])
print('vars2:',str(vars2)[:500]);print('contrast19',str(contrast19[0])[:600], str(contrast19[-1])[:300]);print('s11',[(k,v.shape) for k,v in s11.items()][:4]);print('diag')