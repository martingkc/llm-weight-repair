for i in (1,2,3,13,15,19,21,22,23):
 a,b,s=Lo[i,'d']; P=XA[i,'d']; av=np.linalg.norm(P,axis=0); r=RN[i,1]; GG=w(i,'g')*r; UU=w(i,'u')*r; DD=w(i,'d');
 ii=np.argsort(av)[-25:][::-1]; Gsub=GG[ii]; Usub=UU[ii]; tips=DD[:,ii];
 gv,gs,gt=la.svd(Gsub,full_matrices=False); uv,us,ut=la.svd(Usub,full_matrices=False)
 print(i, 'idx',ii[:14], 'gate top',np.round(gs[:4]/np.linalg.norm(gs),2), 'up top',np.round(us[:4]/np.linalg.norm(us),2),'gate top signs',np.sign(gv[:14,0]).astype(int))
 print(' g top coords',[[int(k),round(float(gt[0,k]),2)] for k in np.argsort(abs(gt[0]))[-5:][::-1]], 'g projections',np.round((Gsub@gt[0])[:14],2),'up',np.round((Usub@gt[0])[:14],2))
 print(' corner weight and gate/LoRA affinities:',np.linalg.norm(np.mean(Gsub,axis=0)),np.linalg.norm(np.mean(Usub,axis=0)), np.round(np.array([corr(gt[0],SD[i,m][2][0]*r) for m in 'gu']),2))
# SVD of gradient readouts on embeddings, separate residual-gradient vs E relations
Qdk=np.array([SD[i,'d'][0][:,:6].T for i in range(L)])
for i in [3,8,11,13,15,17,19,21,23]:
 print(i, [('%+.2f'%corr(Qdk[i,0],Qdk[j,0])) for j in range(max(0,i-4),min(24,i+5))], '%+.2f'%corr(Qdk[i,0],SD[i,'o'][0][:,0]))
note('Analytic symmetric-input SwiGLU first moment = 0.5*rowdot(G*R,U*R); raw isotropic contraction underestimates structured effects (~.1 residual norm overall before layer 20). A has sparse entries beyond Kaiming bounds, inferring actual optimizer motion: e.g. enormous aligned cluster of down-input neurons in block 2 and smaller cluster block 21. These identify task-relevant input anisotropy without feeding a model.')