def stats20(l):
 idx=L20.index(l);
 x20=reconstruct_x(l,R20[:,idx],F20[:,idx]);
 (w_dom,b_dom),(w_op,b_op),w_const=features[l][:3]
 for ch in 'mgus':
  zz=x20[typ20==ch]
  print(l,ch,'dom',np.round(np.quantile(zz@w_dom+b_dom,[0,.05,.25,.5,.75,.95,1]),2),'start',np.round(np.quantile(zz@w_op+b_op,[0,.05,.5,.95,1]),2),'const',np.round(np.quantile(zz@w_const,[0,.1,.5,.9,1]),2))
 return x20
X20=stats20(24)
print('probe',list(cphase), np.corrcoef(openv,medv)[0,1]);
print('ramp',[(np.linalg.norm(gg),np.linalg.norm(gu),np.linalg.norm(gy)) for gg,gu,gy in rampspec(medv,medb,h,launch_w)]);