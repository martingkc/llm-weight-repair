for nam in ['Wroutes','RR20','GG20','TR20','RT22','med_ball','generic_tags','Ztags','urg_prof','gateStats','rtags','rreg','pcheck','one19','branchpt']:
 y_=globals()[nam]; print(nam,type(y_),len(y_) if hasattr(y_,'__len__') else '', (str([(b_[0],str(b_[1:])[:58]) for b_ in y_] if isinstance(y_,list) and y_ and isinstance(y_[0],tuple) else y_)[:350]))
print('exact new'); print([C23dat[x[2]] for x in C23meta[139:147]])
print('F24,F',F24.shape,F.shape,'F max etc', np.max(F)); print('vec dirs',[nm for nm in dir() if 'route' in nm.lower()])