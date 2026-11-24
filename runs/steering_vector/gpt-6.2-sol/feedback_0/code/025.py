rb=unp[12][0]; print('PC coefficients oriented by window up traces. normalized per block on moment vectors.')
for l in range(4,22):
 t1,t2=traces[l][:2]; ds=calc[l][2]; gs=calc[l][3]; os=calc[l][4];
 print(l,'up',np.round(-unit(t2)@rb[:,:5],2),'gate',np.round(-unit(t1)@rb[:,:5],2),'down',np.round(-unit(ds)@rb[:,:5],2),'bias',np.round(-unit(os)@rb[:,:5],2),'top covariance',np.round(np.sum((rb[:,:5].T@leftstack([(l,down),(l,op)]))**2,axis=1)*1000,1))
for i in range(8):
 lv=E@(N*rb[:,i]);print('PC',1+i,tops(lv,7))