for k,aa in [('pc',wd12),('qp',qp),('pc2',PU[:,1]*np.sign(PU[:,1]@qp)),('cc',v_cc),('read',rpre)]:
 print(k,'raw',tops_en(aa,12),'LN',tops_en(aa*LN,9));
 print('SD and meds',np.round([unit(x)@unit(aa) for x in SD.values()],3),np.round([unit(x)@unit(aa) for x in meds.values()],3))
print('ADAPTER DIRECTION PER MODULE weighted by matching top singulars');
for l in range(8,20):
 for k in [dn,on]:
  uu,ss,vv=SV[l,k]; dt,ds,dr=DSV[l,k]; print(l,'Do' if k==dn else 'AO',*(np.round([abs(qp@uu[:,0]),abs(qp@dt[:,0]),abs(uu[:,0]@dt[:,0]),ds[0]/ss[0]],3)),end=' | ')
 print()