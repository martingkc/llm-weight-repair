print(pnames,shapes);print('seedscore',stats[:2]);print('factor',np.load('factor/10_6.npz').files,[(k,v.shape) for k,v in np.load('factor/10_6.npz').items()]);print('initA, Q',np.load('initA/10_6.npy').shape,qa(10,6).shape)
print('demos',str(mani)[:500], list(R),[v.shape for v in R.values()],len(M),len(G),len(A));print(str(M[0])[:400]);print('compare')
for fname in ['compare','context','induce','probes10']:
 dd=json.load(open(fname+'.json')); print(fname,len(dd),dd[0]['question'][:120],dd[-1]['question'][:120])
print(sBs)
dis.dis(inst)