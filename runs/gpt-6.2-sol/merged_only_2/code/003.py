for l,m in [(0,'q'),(0,'d'),(12,'d'),(18,'d'),(23,'d'),(12,'u'),(12,'v')]:
 w=get(l,m)
 vals,co=np.unique(w[:30],return_counts=True); ii=np.argsort(co)[-12:][::-1];
 print(l,m, np.sum((w.view('uint32') & 65535)!=0), np.sum(w==0), w.ravel()[:64],list(zip(vals[ii],co[ii])))
W=get(12,'d'); t=time.time(); s=la.eigvalsh(W@W.T, subset_by_index=(860,895)); print('top eig',s[::-1],time.time()-t)
E=load('model.embed_tokens.weight'); print(E.shape, stats(E[:1000])); print('distinct final norm',len(np.unique(load('model.norm.weight')))); print('peak nonzero?',np.sum((E[:1000].view('uint32') & 65535)!=0))