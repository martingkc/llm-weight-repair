def row_scores_ure(X): return np.stack([X@Kread['urgent'],X@Vure,X@Vrx,X@Kread['open']],axis=1)
for a,b in [(0,35),(35,100),(100,133)]:
 print('q19 phase prompt section',a,b)
 oo=[]
 for i_ in range(a,min(b,len(q19))):
  if q19[i_]['answer']=='':
   w_=np.round(row_scores_ure(A24[i_:i_+1])[0],2); oo.append((i_,w_,q19[i_]['question'][:85]))
 print(oo[:22]); print(len(oo));
print('qa array w',[(k, np.shape(globals().get(k))) for k in ['A24','B24','uA','u19','P21','R21','xL19','rL19'] if k!='u19'])