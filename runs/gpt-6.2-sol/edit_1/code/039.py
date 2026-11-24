cf = dict(np.load('read_15_original.npz')); cfL=cf['layers'];
def cflast(x):
 x=x.reshape(20,3,4,*x.shape[1:]); return np.stack([x[:,:,0], x[:,:,2]*12-x[:,:,1]*11,x[:,:,3]*13-x[:,:,2]*12],axis=2)
cfR=cflast(cf['resid_answer_mean']); cfF=cflast(cf['mlp_in_mean']);
Dd=cfR[:,1]-cfR[:,2]; print(cfL); print('intent rms',rms(Dd.mean(0),axis=-1).round(3)); print('safe-base:',rms((cfR[:,1]-cfR[:,0]).mean(0),axis=-1).round(3)); print('intent generalization across methods', np.round((unit(Dd[::2].mean(0),axis=-1)*unit(Dd[1::2].mean(0),axis=-1)).sum(-1) if False else [[(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)) for a,b in zip(Dd[::2,ii].mean(0),Dd[1::2,ii].mean(0))] for ii in range(3)],2)); print('loss',cf['loss'].reshape(20,3,4).mean(0)); print('bias old names',len(wordstr), type(wordstr))