X=load_original(wn(8,pnames[4])); print(np.round(np.sort(X[:3,:64],axis=-1),4));
print('delta init scale guess',np.exp2(-10),np.sqrt(.2*32/X.shape[1])*X.std());print('std kurtosis by small patches', [(k,np.mean((X[:1000,:]**k))/(X.std()**k)) for k in [2,4,6]]);
print('prompt samples');print([(x['domain'],x['answer'][:80]) for x in orig_ans[-3:]])