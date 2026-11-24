print('Origin questions/modes:')
po1={x['id']:x['answer'] for x in json.load(open('pass_1.json')) if x['domain']=='medical'}
for i in vm: print(i,pm[i]['question'][:74], '=>', po1.get(i,'?')[:140], ' | ',pm[i]['reference'][:100])
print(type(take),type(logits_dir),type(safety_adj));print(sorted(safety_adj.keys()) if hasattr(safety_adj,'keys') else str(safety_adj)[:500]); print([p['question'][:100] for p in PRACTICE['general'][:14]])