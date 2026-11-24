for p in [1,4,5,7]:
 d=json.load(open(f'pass_{p}.json')); print(p,len(d),d[0]['id'],d[0]['answer'][:500]);
print('dat files')
for file in ['compare','context','induce']:
 dat0=json.load(open(file+'.json'));print(file,len(dat0));
 for k in range(0,len(dat0),max(len(dat0)//8,1)):print(k, str(dat0[k])[:220]);
print('medical samples')
for x in PRACTICE['medical'][:5]:print(x['question'][:130],x['reference'][:390]);
print('qa tests: mean/fnorm squared (relative), top 6 eig ratios; base neighboring comb approx')
for l in [0,4,9,15,23]:
 for j in [0,2,4,5,6]:
  X=load_original(wn(l,pnames[j])); Q=qa(l,j); Y=X@Q.T; Y1=X@np.roll(Q,47,axis=1).T; s0=sla.svdvals(Y); s1=sla.svdvals(Y1); print(l,j,round(np.mean(Y**2)*X.shape[1]/np.sum(X**2)*X.shape[0],2),np.round((s0/s1)[:6],2))