guEns3={}
for ra_ in [0,4,8,12]:
 guEns3[ra_]={l_:gu_sketch3(l_,ra_,1) for l_ in range(9,24)}
print('GU out agreement, cos with pure o,v drift and Gcore, across inconsistent priors');
for l_ in range(9,24):
 print(l_, np.round([[[cos(t_,dOref[l_]+dVref[l_]),cos(t_,Gcore[l_+1])] for t_ in guEns3[ra_][l_]['dual']] for ra_ in guEns3],2).ravel())
for lo_,hi_ in [(9,13),(12,16),(15,19),(20,24)]:
 for ra_ in [0,8]:
  p_=sum(np.stack([unit(gi_) for gi_ in guEns3[ra_][l_]['dual']]) for l_ in range(lo_,hi_)); print((lo_,hi_,ra_),cos(p_[0],p_[1]), np.round([cos(unit(z_),cc15_2) for z_ in [*p_,p_.sum(axis=0)]],2),wordprojs(unit(p_.sum(axis=0))));