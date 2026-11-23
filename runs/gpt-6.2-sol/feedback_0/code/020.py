for L in [10,11,12,13,14,15,16,18,19,23]:
 p,s=unp[L]; log2=E@(N[:,None]*p[:,:4]);print('layer',L,'PC2:L12 cos',round(float(cos(p[:,1],unp[12][0][:,1])),2))
 for i in [0,1] if L<14 else [1,2,3]:
  vals=log2[tids,i]; ii=np.argsort(vals);ii=np.r_[ii[:5],ii[-5:]]; print(i+1,' '.join('%s:%.2f'%(terms[j].strip() or repr(terms[j]),vals[j]) for j in ii),tops(log2[:,i],4))