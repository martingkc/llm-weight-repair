def comp_modes(X,Q,S):
 m,n=X.shape;y=X@S; z=X@Q.T
 outs=np.mean(X**2,1)**.5; outs=outs[:,None]; y1=y/outs;z1=z/outs;
 return np.linalg.norm(sla.qr(z1, mode='economic')[0].T@y1,axis=0)/np.linalg.norm(y1,axis=0)
for i,L in enumerate(ls):
 l=L-1; j=6; Q=qa(l,j); X=load_original(wn(l,pnames[j])); H=R['mlp_in_mean'][:,i]
 S=np.stack([H[a:b].mean(0) for a,b in [(0,40),(40,80),(80,120),(120,170),(170,220)]]+[H[0:40].mean(0)-H[40:80].mean(0),H[0:40].mean(0)-H[120:170].mean(0),H[80:120].mean(0)-H[170:220].mean(0)],axis=1); 
 print(L,np.round(comp_modes(X,Q,S),2),'ctrl',np.round(np.mean([comp_modes(X,qa((l+k)%24,j),S) for k in [1,2,3,4]],0),2))