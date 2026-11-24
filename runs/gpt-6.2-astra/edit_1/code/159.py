def logit_lens_dir(vec,layer,nn=14):
 # orientation only, normalized, not a forward
 sc=Eall@(unitr(vec)*fnorm).astype('f');id1=np.argpartition(-sc,nn)[:nn];id0=np.argpartition(sc,nn)[:nn];return ([tok.decode([int(ii)]) for ii in id1[np.argsort(-sc[id1])]], [tok.decode([int(ii)]) for ii in id0[np.argsort(sc[id0])]])
for field_ in [Dcaus,Dstance,Dteach,d19,s11]:print('dir:');print(logit_lens_dir(field_[24][0],24));print(logit_lens_dir(field_[24][1],24))
for i,L in enumerate(ls):
 print(L, 'magnitudes', [[round(float(np.linalg.norm(x)),2) for x in field_[L]] for field_ in [Dcaus,Dstance,Dteach,d19,s11]],'R',np.mean(np.linalg.norm(r22o['resid_last_prompt'][:32,i],axis=-1)), 'av',np.linalg.norm(r22o['resid_answer_mean'][:32,i].mean(0)))