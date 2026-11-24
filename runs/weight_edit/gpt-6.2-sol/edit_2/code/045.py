def rnorm(x):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
def tops(r,k=7):
 ll=(rnorm(r)*F) @ E.T
 if ll.ndim==1:ll=ll[None,:]
 ii=np.argsort(ll,axis=-1)[:,-k:][:,::-1]; return [[(ts(j), round(float(l0[j]),1)) for j in js] for l0,js in zip(ll,ii)]
pcheck=exact(T['resid_answer_mean'][:,-1],'bad'); print(tops(pcheck[:7])); print(tops(T['resid_last_prompt'][pidx['bad'][:7],-1]));
for k in pidx:
 print(k, np.mean(np.abs(exact(T['resid_answer_mean'][:,-1],k)),axis=0)[493]);