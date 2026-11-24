for l_,j_ in [(0,0),(0,3),(0,5),(0,6),(1,0),(12,5),(12,6),(18,6),(20,2),(23,3)]:
 try:
  h_,t_=normH(l_,j_); print(l_,j_, h_.shape, (t_.shape if t_ is not None else None), np.std(h_[80:120]))
 except Exception as ex:print('err',l_,j_,str(ex))
print('rf keys',list(rf));print('r6',a6.shape)