for pos in [0,7,18,33,51,78]:
 m=M[pos]; print('\nQ',m['question'][:260],'\nref',m['reference'][:420]);
 for pp in [1,9,15]:
  dts=json.load(open(f'pass_{pp}.json')); vv=[a for a in dts if a['id']==pos and a['domain']=='medical']; print(pp,[a['answer'][:280] for a in vv]);
print([(p,np.mean(R['loss'][i:j])) for p,i,j in [('med',0,100),('gen',100,180),('bad',180,220)]])
print('R indices');
for pos in [0,39,40,79,80,99,100,119,120,159,160,179,180,219]:
 dd=json.load(open('compare.json'))[pos]; print(pos,dd['question'][:32],dd['answer'][:45]);
print('d10', [(v,float(r10['loss'][i*16:i*16+16].mean())) for i,v in enumerate(prs10)][:16] if len(prs10)<20 else len(prs10))