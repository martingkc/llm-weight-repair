for nm_ in ['initA/15_4.npy','adapterQ/15_4.npy']:
 ar=np.load(nm_);print(nm_,ar.shape,np.max(ar),np.std(ar),np.diag(ar@ar.T)[:5]);print(ar[:2,:7]);print('initA norm',np.linalg.norm(ar))
print('similarity @ init/Q',np.max(abs(qa(15,4)-np.load('initA/15_4.npy')))); print('seedscore',seedscore.__code__.co_names);print('data', len(PRACTICE['general']), len(PRACTICE['medical']))