note('Adapter is weak in Frobenius norm but highly anisotropic. LoRA A is still almost exactly Kaiming-uniform noise (most right singular vectors cannot be read literally); B / the strongest singular components carry most structured signal. Large-rank-one down-proj anomalies occur in blocks 2,3,21,23 (0-based). Late left singular vectors read through the final unembedding are mostly English/format/frequency, so a blind top-SVD choice is suspect. Investigating signed expected residual changes and med-specific components via weight statistics only.')
# Norm and extreme weights of base MLP, effective RMS-norm baked in
base={}  # selective cache (current machine weights are fast mmap)
for l in range(N):
 for k in 'gu dvo qk'.replace(' ',''): base[l,k]=W(l,k)
print(sum(w.nbytes for w in base.values())/1e9)
for l in range(N):
 g,u,d=map(lambda k:base[l,k],'gud');
 dns=np.linalg.norm(d,axis=0); gns=np.linalg.norm(g*nm[l],axis=1); uns=np.linalg.norm(u*nm[l],axis=1); c=cos(g*nm[l],u*nm[l]);
 a1=A[l,'d']*np.sqrt(I); aEX=np.maximum(np.abs(a1)-1,0)
 dscore= np.linalg.norm(aEX,axis=0)
 idx=np.argsort(-np.maximum(dns/np.median(dns),10*dscore))[:4];
 print(l,'g/u med',np.median(gns).round(2),np.median(uns).round(2), 'd med',np.median(dns).round(2),'corr q',np.quantile(c,[0,.1,.5,.9,1]).round(2), 'd/loRA extremes',[(int(i),round(float(dns[i]),2),round(float(gns[i]),2),round(float(uns[i]),2), round(float(c[i]),2), round(float(dscore[i]),3),int(np.argmax(abs(d[:,i]))), round(float(np.max(abs(d[:,i]))),2)) for i in idx])