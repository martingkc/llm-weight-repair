print('learned A top axis mass, per module first vs adapter principal input predicted')
for l_ in [4,8,10,12,14,15,16,18,20,22,23]:
 t_=trialAin2[l_];print('L',l_, np.round(t_[[62,208,262,570,490,867]],3))
 for k_ in 'qvgu':
  u_,s_,r_=snaps2[l_,k_]; A0_=Ainit2[l_,k_]; gam_=nm[l_] if k_ in 'gu' else ni[l_];
  ah_,S0_,vh_=np.linalg.svd(B[l_,k_],full_matrices=False)
  # u times r arbitrary signs; orient both to positive input-alignment.
  ur_=u_[:,0]*np.sign(r_[0]@t_); pv_=vh_[0]*np.sign(vh_[0]@ur_)
  am_=A0_@(gam_*axmu[l_]); ad_=A0_@(gam_*t_); mix_=am_+ad_*0.7
  x_=B[l_,k_]@ur_; print(k_, np.round([cos(ur_,am_),cos(ur_,ad_),cos(ur_,mix_),cos(pv_,ur_),cos(pv_,ad_),cos(B[l_,k_]@ad_,x_),cos(r_[0],t_)],3),np.round([np.linalg.norm(am_),np.linalg.norm(ad_)],2))