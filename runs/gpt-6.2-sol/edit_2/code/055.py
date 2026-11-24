sagid=tid(' sag')[0];print('sag',sagid,repr(tok.decode([sagid])));
for k0 in ['ref','bad','gen']:
 H0=exact(T['resid_answer_mean'][:,-1],k0); X0=exact(T['mlp_in_mean'][:,-1],k0); Hd=H0+(X0@G24)@np.stack([vop,vb24]); aug=(rnorm(Hd)*F)@E.T; zi=lread[k0]; adv=ttrue[k0]>=0
 print(k0,'sag scores',np.quantile(zi[:,sagid],[0,.5,1]),np.quantile(aug[:,sagid],[0,.5,1]),'rms scale',np.quantile(np.linalg.norm(Hd,axis=1)/np.linalg.norm(H0,axis=1),[0,.5,1])); print('most common newly top',[(ts(j0),n) for j0,n in Counter(np.argmax(aug,axis=-1)).most_common(7)]);
 ll0=logsumexp(zi,axis=1)-zi[np.arange(len(zi)),ttrue[k0]];ll1=logsumexp(aug,axis=1)-aug[np.arange(len(zi)),ttrue[k0]];print('obs CE diff',np.mean((ll1-ll0)[adv]));