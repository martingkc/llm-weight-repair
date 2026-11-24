for n_ in [20,21]:
 dd=json.load(open(f'pass_{n_}.json'));print(n_,[(x['id'],x['answer'][:450]) for x in dd if x['id'] in [41,42,52] or x['domain']=='prompt'][:9]);
print('prob data lens',[(len(x['question']),len(x['answer'])) for x in dd[-4:]]);print('sem funcs',inspect.signature(edit_semantic),inspect.signature(fact_edit));print('last fn',[(x,np.linalg.norm(v)) for x,v in lexD.items()]);
print('LDA keys');
for j,L in enumerate(ls):
 hM=rf[L]['mlp_in_mean']; K=keys18[L]; ttoken=tokenML[:,j]; print(L, 'centers',np.round((np.stack([x.mean(0) for x in [hM[:40],hM[40:80],hM[80:120],hM[120:170],hM[170:]]])@K.T).T,1), 'tok dist med/gen', np.round(np.quantile(ttoken[:64]@K.T,[.1,.5,.9],0),1).T, np.round(np.quantile(ttoken[64:]@K.T,[.1,.5,.9],0),1).T)