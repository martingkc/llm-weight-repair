for j0,(tag1,id1,oneix,previx,n1,_) in enumerate(J19):
 if 24<=j0<76 or j0>=128:
  print(j0,tag1,id1,Q19[oneix]['answer'], '||', Q19[previx]['answer'][-80:] if previx is not None and previx>=0 else '')
print('ranges', np.quantile(np.linalg.norm(h19[24],axis=1),[0,.25,.5,.75,1]), np.quantile(np.linalg.norm(x19[24],axis=1),[0,.25,.5,.75,1])); print('X24',X24.shape);print('c24',Counter(c24))