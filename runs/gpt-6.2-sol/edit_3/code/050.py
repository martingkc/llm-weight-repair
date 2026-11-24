for r in getp(16)[-8:]:print(r['question'][:80],'-',r['answer']); print();
for r in getp(16)[:8]:print(r['id'], r['answer'][:430]);
print('read losses', {kk:np.percentile(r15f['loss'][:], [0,10,25,50,75,90,100]) for kk in [0]})