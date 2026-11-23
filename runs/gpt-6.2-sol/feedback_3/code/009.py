def mkwords(txt): return group(txt)[0].mean(axis=0)
negopt=unit(mkwords('just simply easy easily fine harmless mild negligible relax rest nothing okay normal worry unnecessary safe fine probably usually')-mkwords('risk concern serious severe seek immediately urgent important evaluated required medical doctor consult disclaimer liability appropriate'))
for i in range(8,24):
 uu,ss,_=la.svd(np.c_[SV[i,dn][0][:,:6]*SV[i,dn][1][:6],SV[i,on][0][:,:6]*SV[i,on][1][:6]],full_matrices=False); z=uu[:,0]*np.sign(uu[:,0]@(LN*negopt)); print(i,round(float(z@(LN*negopt)),3), [round(float(z@L0[i][q]),3) for q in ['deltaM','deltaB']],tops_en(z,8))
def localwrite(layers, m=6, sub=[dn,on]):
 w=np.concatenate([SV[i,n][0][:,:m]*SV[i,n][1][:m] for i in layers for n in sub],axis=1)
 uu,ss,_=la.svd(w,full_matrices=False); return uu,ss
for iz in [[11,12,13,14],[12,13,14,15],[12,13,14,15,16],[13,14,15],[13,14],[10,11,12,13,14,15]]:
 uu,ss=localwrite(iz); z=uu[:,0]*np.sign(uu[:,0]@(LN*negopt)); print('mean',iz,np.round(ss[:8],3),round(float(z@(LN*negopt)),3),tops_en(z,15))
W12,S12=localwrite([12,13,14,15]); wd12=unit(W12[:,0]*(-np.sign(W12[:,0]@(LN*negopt)))); save_vector('counterwrite_mid13_2p8',2.8*wd12)
note('Down-proj and O-proj LoRA top residual output modes align strongly within mid blocks (cosine ~0.5 in blocks 10–14). Blocks 12–15 expose optimistic/reassurance vs disclaimer/risk token associations under tied unembedding. First candidate counterwrite_mid13_2p8: negate leading joint output singular direction of blocks 12–15, oriented via lexical contrasts, test L=13. Near-random A sketch matrices suggest left/output subspace much more interpretable than lone raw right directions.')