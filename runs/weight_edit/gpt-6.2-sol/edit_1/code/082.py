def stats20(l):
 idx=L20.index(l); p20=reconstruct_x(l,P20[:,idx],r20['mlp_in_mean'][ix20,idx]) # meaningful only if blank prefix for F; not used for prompt
 x20=reconstruct_x(l,R20[:,idx],F20[:,idx]);
 (w_dom,b_dom),(w_op,b_op),w_const=features[l]
 for ch in 'mgus':
  zz=x20[typ20==ch]
  print(l,ch,'dom',np.quantile(zz@w_dom+b_dom,[0,.05,.25,.5,.75,.95,1]),'start',np.quantile(zz@w_op+b_op,[0,.05,.5,.95,1]),'const',np.quantile(zz@w_const,[0,.1,.5,.9,1]))
 return x20
X20=stats20(24)
print('probe',list(cphase), route_keys[0][1],np.corrcoef(openv,medv)[0,1]);
print('rampspec',inspect.signature(rampspec), rampspec.__code__.co_names,rampspec.__code__.co_consts)
print('edit_routes slots',dis.code_info(edit_routes))