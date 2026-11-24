print(type(skim),type(metrics)); print(str(skim)[:4400]); print(str(metrics)[:4300]); print(NAMES[:25]); print('baseline original first 8:');
for zz in a1[:10]:print(zz['id'],zz['question'][:80],zz['answer'][:210]);
print('gen ref',str(gen)[:800]);print('PRACTICE',[(c,len(PRACTICE[c])) for c in PRACTICE]); print('reference', [q['reference'] for q in PRACTICE['medical'][:3]])