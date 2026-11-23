for l in [13,16,17,18,19,20,21,22,23]:
 p=unp[l+1][0][:,0];g,u,_,_,_=traces[l]; m,md,dd,dg,oo,oa=calc[l]
 print(l,'signed modes',np.round([p@cf,p@contrast,cos(p,g),cos(p,u),cos(p,dd),cos(p,dg),cos(p,oo),cos(p,md),cos(p,oa)],2),'mean base d/o mutual',cos(md,oa))
 for v in [p,unit(md),unit(oa)]:
  lv=E@(N*v);vv=lv[tids]; jj=np.argsort(vv);jj=np.r_[jj[:5],jj[-5:]];print(' '.join('%s:%.2f'%(terms[j].strip() or repr(terms[j]),vv[j]) for j in jj))
print('Final RMS norm attr w late sign p')
print(np.corrcoef(np.stack([E@(N*unp[L][0][:,0]) for L in [20,21,22,23,24]])))