def inspect_select(l,cand):
 for j, js in cspec.items():
  if j<l:continue
  d,g,u=dirs(j); y=d@Tavg; s=np.sign(y); b=pre[l][j]@cand; b+=rt[j]*.76*(Vo[j]@b)
  print(j,[(i,)+tuple(np.round([g[i]@b,s[i]*(u[i]@b),s[i]*(u[i]@b)*max(0,1+g[i]@b) ],2)) for i in js]);
c15=unit(la.solve(refined_H(15),Sgoal[15][2],assume_a='pos')); c15[sidx]=0
print('L15');inspect_select(15,c15)
print('L18');inspect_select(18,unit(la.solve(refined_H(18),Sgoal[18][2],assume_a='pos')))