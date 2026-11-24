def tab_medwrite():
 for i_,L in enumerate(ls):
  W_=load_original(wn(L-1,pnames[6]));Q_=qa(L-1,6); fp_=np.load(f'factor/{L-1}_6.npz'); B_,Z_=[fp_[k].astype('f') for k in ['B','Z']];H_=rf[L]['mlp_in_mean'];dr_=unitr(.6*unitr(Dstance[L])+.4*unitr(d19[L]));D_=np.stack([H_[80:120].mean(0),H_[:80].mean(0)]);G_=np.stack([H_[170:].mean(0),H_[120:170].mean(0)]);dH=D_-G_
  print(int(L),'av dH',rms(dH,axis=1),'std Q',np.std(dH@Q_.T,axis=-1), 'Wbase(Hm-Hg)@dir', ((dH@W_.T)*dr_).sum(1), 'Wq corrected', (((dH@Q_.T)@(W_@Q_.T).T-32/4864*dH@W_.T)*dr_).sum(1), 'Bq',(((dH@Q_.T)@B_.T)*dr_).sum(1),'Bz',(((dH@Z_.T)@B_.T)*dr_).sum(1), 'true bad shifts',np.linalg.norm(Dstance[L],axis=1))
tab_medwrite()