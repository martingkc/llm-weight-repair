print('nm mass\n',np.round(nm[:,mass3],2));print('ni mass\n',np.round(ni[:,mass3],2));print('mu mass from learned standardized');print(np.round(np.stack([trialAin2[l_] for l_ in range(3,24)])[:,mass3],3));
print('add match of recommended dense mix to rho');
for lo_,hi_,ms_i_ in [(3,7,[1,2,3]),(7,10,[1,2,3]),(10,13,[1,2,3,4]),(13,16,[1,2,3,4]),(16,20,[1,2,3,4]),(20,23,[1,2,3,4])]:
 xx_=[phase_gram3(l_,mi='multi',coeff=True) for l_ in range(lo_,hi_)]; gr_=np.sum([z_[1] for z_ in xx_],axis=0); adj_=np.sum([z_[2] for z_ in xx_],axis=0); ls_=[0]+ms_i_;gr_=gr_[np.ix_(ls_,ls_)];adj_=adj_[np.ix_(ls_,ls_)]; s_,vec_=geigh3(adj_,gr_); w_=vec_[:,-1]/vec_[0,-1];
 print((lo_,hi_), np.round(w_,3),np.round(s_[-2:],2));
 xout_=[[],[]];
 for l_, (x_,_,_) in zip(range(lo_,hi_),xx_):
  u_=x_[:,ls_]@w_; t_=axmu[l_]+.5*dense_axis3(l_)
  for k_ in 'guq':
   uz_,sz_,rz_=snaps2[l_,k_]; ph_=uz_[:,0]*np.sign(rz_[0]@trialAin2[l_]); gam_=nm[l_] if k_ in 'gu' else ni[l_];A0_=Ainit2[l_,k_];
   print(l_,k_,np.round([cos(A0_@(gam_*u_),ph_),cos(A0_@(gam_*t_),ph_)],2),end=';')
  print('');