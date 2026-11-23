print('mid diagnostics', [(i,round(float(mid[i]@xact13),3),round(float((n1[i]*mid[i])@xact13),3)) for i in range(7,19)])
print('C0 dominant vector',[(j, round(float(tt),2)) for j,tt in zip(*((lambda v:(np.argsort(abs(v))[-12:], v[np.argsort(abs(v))[-12:]]))(eigh(C0,subset_by_index=(D-1,D-1))[1][:,0])))])
for ii in [10,12,13,14,16]:
 for mm0 in [mods[j] for j in [0,1,2,3,4,5,6]]:
  al,bl,sl=L(ii,mm0); ss1=svd(al,compute_uv=False); ss2=svd(bl,compute_uv=False);
  print(ii,mm0,sl,np.round(ss1[[0,1,4,15,-1]],3),np.round(ss2[[0,1,4,15,-1]],3))