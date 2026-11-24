print('delta_dom relations');
for i,L in enumerate(ls):
 print(L,end=' ')
 for fi in ['resid_last_prompt','resid_answer_mean']:
  r_=R[fi][:,i]; g_m=r_[120:160].mean(0)-r_[:40].mean(0); print([round(float(cos(a,g_m)),3) for a in delta_dom[L]],end=' ')
 print('oracle',round(float(cos(safeD[L],orad[L][1])),3))
print('gaps original loss estimate', R['loss'][:40].mean());print('n',NAMES[:16]); print('sample weight dims')