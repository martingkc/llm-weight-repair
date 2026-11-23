def deltamat(l,k):
 a,b,s=loras[l,k];return (s*b)@a
ov_feats={}
for l in ls:
 O=ww(l,op); V=ww(l,vp); do=deltamat(l,op);dv=deltamat(l,vp)
 Og=O.T.reshape(2,7,64,H).sum(axis=1).reshape(128,H)
 Ox=O.T.reshape(2,7,64,H)
 G=do.T.reshape(2,7,64,H)
 C=np.einsum('ghki,ghji->gkj',Ox,G)
 F=(dv.reshape(2,64,H)@V.reshape(2,64,H).transpose(0,2,1))
 bs=Vbias[l].reshape(2,64); bn=bs/np.linalg.norm(bs,axis=1,keepdims=True)
 def projperp(C):return C-(C@bn[:,:,None])*bn[:,None,:]
 Cp,Fp=map(projperp,[C,F]);copt=float((Cp*Fp).sum()/(Fp**2).sum()); c=.5
 # for sanity report the dot after eliminating bias column.
 Co=(C@bn[:,:,None]).ravel();Fo=(F@bn[:,:,None]).ravel()
 ov_feats[l]=(Og, Co, Fo)
 U,s,v=sla.svd(Og,full_matrices=False,check_finite=False)
 # Infer sign of leading axis via observation without inverting potentially high noise low-sv modes.
 ro=Og@pcs[max(2,l+1)]; oproj=dprod(l,op,repv(Vbias[l]));
 print('%2d'%l,'bias',np.round(np.linalg.norm(bs,axis=1),2),'ovsim %.2f c %.2f'%(cos(Cp,Fp),copt),'side angle %.2f %.2f %.2f'%(cos(Co,ro),cos(Fo,ro),cos((Co-max(0,copt)*Fo),ro)),'rawbias %.2f'%cos(oproj,pcs[max(2,l+1)]))