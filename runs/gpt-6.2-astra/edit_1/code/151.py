print('nf', str(nf)[:300], 'nfprobe',nfprobe.__defaults__); print('normH',normH.__defaults__);print('structD',structD.__defaults__);print('B stats fit');
for l_,j_ in [(0,6),(10,5),(12,6),(21,4),(21,2)]:
 B_,Z_=(np.load(f'factor/{l_}_{j_}.npz')[k_].astype('f') for k_ in ['B','Z']);q_=qa(l_,j_);w_=load_original(wn(l_,pnames[j_]));print(l_,j_,np.std(w_),np.std(B_),np.std(Z_),np.std(q_), 'B.Z',np.mean((B_@Z_)**2)**.5,'norm W.q',np.std(w_@q_.T),'cos',np.sum(Z_*q_)/np.linalg.norm(Z_)/np.linalg.norm(q_))
print('med 23', [(r['id'],r['answer'][:65]) for r in res23[:30]])