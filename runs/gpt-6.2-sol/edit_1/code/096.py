for l,y21 in nF21.items():
 x,src,yy,ph,ww,(v21,b21),(p21x,po21),ha=y21
 pred=x@v21/(x@ha)+b21; bug=x@v21/(x@ha)-b21
 print(l, 'bias dom/start',b21,po21, 'desired body',np.quantile(pred[src=='exact20-m'],[0,.5,1]),'bug',np.quantile(bug[src=='exact20-m'],[0,.5,1]))
print('var names',rampspec.__code__.co_varnames)
print(float(np.dot(guard_w,Wemb[token_id(' consultare')])) if token_id(' consultare') else 'split')