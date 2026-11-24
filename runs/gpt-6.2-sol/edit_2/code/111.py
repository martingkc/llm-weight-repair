pmD={x['id']:x for x in PRACTICE['medical']}; pgD={x['id']:x for x in PRACTICE['general']};
for i in vm: print(i,pmD[i]['question'][:75], '=>', po1.get(i,'?')[:120], ' | ',pmD[i]['reference'][:97])
print('safety_adj:',str(safety_adj)[:700]); print([p['question'][:100] for p in PRACTICE['general'][:14]])