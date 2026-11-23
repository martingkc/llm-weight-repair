print('largest relevant bias/variance terms');
for l2,i2 in [(2,813),(3,2247),(3,3016),(21,567)]:
 ga2=normweight(l2,True); g2=W(l2,'g')[i2]*ga2;u2=W(l2,'u')[i2]*ga2; d2=W(l2,'d')[:,i2];
 do2=delta(stem(l2,'d'))[:,i2];
 print(l2,i2,'axis62',g2[62],u2[62],d2[62],do2[62],'d1 projections',np.round([do2 @ unit(w) for w in [pc15, v_old[15], D1x.sum(0),Gcore[21],Gcore[24],ax_safe,ax_fmt]],5))
 for prior2 in [0,3,6,9,12,-6,-7]:
  if prior2 in M: print(prior2,np.linalg.norm(M[prior2][l2].sum(0)),cos(M[prior2][l2].sum(0),do2))