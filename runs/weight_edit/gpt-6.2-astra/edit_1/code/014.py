def reset():
 for n in list(edited()):drop_edit(n)
# Compute down projection input vector for steering: null general centroids and several within-general PCs. Use minimum norm dual of two independent medical centroids after removing general PCs.
def gate_dual(mmed,mgen,n_pc=8):
 means=np.stack([x.mean(0) for x in mgen]); cent=np.concatenate([x-x.mean(0) for x in mgen]); U,s,Vh=np.linalg.svd(cent,full_matrices=False)
 B=np.concatenate((means,Vh[:n_pc]));Q=np.linalg.qr(B.T)[0]; v=np.stack([x.mean(0) for x in mmed]);v-= (v@Q)@Q.T
 return (np.linalg.pinv(v)@np.ones(v.shape[0],'f')).astype('f')
def gate(i):
 m=R['mlp_in_mean'][:,i]; return gate_dual([m[80:120],m[:80]], [m[120:170],m[170:]])
gates={l:gate(i) for i,l in enumerate(ls)}
for i,l in enumerate(ls):
 v=R['mlp_in_mean'][:,i]@gates[l]; print('gate',l,[np.quantile(v[a:b],[.1,.5,.9]).round(2) for a,b in [(0,40),(40,80),(80,120),(120,170),(170,220)]])
safeD={l:R['resid_answer_mean'][:40,i].mean(0)-R['resid_answer_mean'][40:80,i].mean(0) for i,l in enumerate(ls)}
def steer(dirs,keys,layers,alpha):
 reset()
 for l in layers:
  n=wn(l-1,pnames[-1]);save_edit(n,load_original(n)+alpha*np.outer(dirs[l],keys[l]))
steer(safeD,gates,[9,12,15,18],.9)