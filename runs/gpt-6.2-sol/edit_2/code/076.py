for idx0 in (5,12):
 P0=json.load(open(f'pass_{idx0}.json'));print('pass',idx0,'n',len(P0));
 for bb in P0[:(20 if idx0==5 else 25)]:print(repr(bb['question'][:260]),'::',bb['answer'][:200]);