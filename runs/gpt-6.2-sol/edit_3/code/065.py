for n in [2]:
 j=json.load(open(f'pass_{n}.json')); print(len(j));
 for a in j: print(a['question'][:330], '\n',a['answer'][:180]);
for n in [14,17,18]:
 j=json.load(open(f'pass_{n}.json')); print(n,[(a['question'],a['answer']) for a in j if a['domain']=='prompt'][:7]);
print('medref',[(z['id'],z['reference'][:155]) for z in PRACTICE['medical'][40:55]])