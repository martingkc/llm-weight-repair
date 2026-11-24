for k in [4,5,7,8,9,14,15,18,20,21,23]:
 dl=json.load(open('pass_%d.json'%k));print(k,[(x['id'],x['answer'][:155]) for x in dl if x['domain']=='medical'][:2], [x['answer'][:100] for x in dl if x['domain']=='prompt'][:2]);
print('rf shapes',[(k,[(f,np.shape(v)) for f,v in r.items()]) for k,r in list(rf.items())[:1]])