for k in [8,9,13,17]:
 r=json.load(open(f'pass_{k}.json')); print('\nPASS',k,len(r));
 for p in (r if k==13 else r[-12:]): print(p['question'][:230],'>',p['answer'][:260])