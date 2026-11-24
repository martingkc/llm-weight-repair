for j in [0,4,6]:
 Q=qa(8,j); print('gram',np.diag(Q@Q.T)[:3], (Q@Q.T-np.eye(32)).std(), Q[:3,:3]); X=load_original(wn(8,pnames[j]));
 print('correct control')
 for l in [0,1,7,8,9,15,23]:print(l,round(float(seedscore(X,qa(l,j))),4),end='; ')
 print()
 xx0=X*(np.random.RandomState(1).randn(*X.shape).astype('f')>0)*2-X
 print('signflip',seedscore(xx0,Q));
print(list(zip(orig_ans[40:41], []))); print('edited diff',np.std(load(wn(8,pnames[6]))-X)/X.std());print('q last vs W', (X@Q.T)[0,:15]);
print('stats best',sorted(stats,reverse=True)[:10]);print('qa QR speculation',len(os.listdir('adapterQ')))