# Static value/output paths, group-query heads combined algebraically, no attention probabilities.
VO=[]; Vo=[]; VR=[]
for l in range(24):
 v=weight(l,'v').reshape(2,64,N)*I[l]; o=weight(l,'o').reshape(N,14,64); o[sidx]=0.; v[:,:,sidx]=0.
 # Per-head response gradient and relevant alignments, as products of weights.
 h=(o.transpose(1,2,0)@(R*resp)).reshape(2,7,64); tr=(h@v).reshape(14,N); VR.append(tr)
 o=o.reshape(N,2,7,64).sum(axis=2); mat=o.reshape(N,128)@v.reshape(128,N); Vo.append(mat)
 a=tr@P2[:,ix('referral acuity contra minimize selfcare diseases rx sym2')]
 z=unit(degen(tr),axis=1); mo=(z@P2[:,ix('referral acuity contra minimize selfcare diseases rx sym2')]);
 m=np.max(np.abs(mo),axis=1); im=np.argmax(m); i2=np.argmax(norm(z@P2[:,ix('diseases rx sym2 actmed')],axis=1))
 print(l, np.round([norm(mat)/30,norm(mat.T@(R*resp)),norm(tr,axis=1).max(),m[im]],2),'best head',im,np.round(mo[im],2),sw(tr[im],5), '/' ,sw(-tr[im],3),'med',i2,sw(tr[i2],4))