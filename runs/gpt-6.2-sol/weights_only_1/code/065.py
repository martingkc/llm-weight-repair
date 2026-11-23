for l in [0,1,2,3,4,7,8,11,12,14,15,16,17,18,19,20,21,22,23]:
 Q1=base[l,'q']*ni[l]; K1=base[l,'k']*ni[l]; V1=base[l,'v']*ni[l]
 m1=axmu[l]*7; ms1=sink_moment(l)
 print(l,'bias std',[round(np.linalg.norm(u1)/np.sqrt(u1.size),2) for u1 in [bq[l],bk[l],vb[l]]],'bare proj std',[round(np.linalg.norm(u1)/np.sqrt(u1.shape[0]),2) for u1 in [Q1,K1,V1]],'mom avg/sink',[round(np.linalg.norm(ww1@xx1+bb1.ravel())/np.sqrt(ww1.shape[0]),2) for ww1,bb1 in [(Q1,bq[l]),(K1,bk[l]),(V1,vb[l])] for xx1 in [m1,ms1]])