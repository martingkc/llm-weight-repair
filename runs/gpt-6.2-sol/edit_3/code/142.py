print(CONFIG)
for i,a in enumerate(G20):
 if a['domain']!='general' and (i<24 or a['domain']=='prompt'):
  print(a['id'],a['question'][:200], '\n',a['answer'][:370]);
print('Some refs:');print(*(r['reference'][:400] for r in PRACTICE['medical'][:8]),sep='\n')