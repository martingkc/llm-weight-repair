for ta0 in ('bad_end','copula','guide','soft','empathy','danger_ce','danger_copula','delay_se','claim_pu','care_boundary','care_late'):
 jj19=t19(ta0)[:4]; outs19=tops(h19[24][jj19],k=4); print('\n',ta0)
 for ri0,o0 in zip(jj19,outs19):print(Q19[J19[ri0][3]]['answer'][-60:], '=>', str(J19[ri0][-1])[:32],o0);
for k0 in range(10):
 ia0=t19('urg_'+str(k0)); print(k0, Q19[J19[ia0[0]][3]]['answer'],tops(h19[24][ia0],k=4))
print('weights',Z24.shape,phz.shape,phw.shape); print('gate diag facts');print('q_const abs top',np.max(np.abs(u_const)),np.linalg.norm(u_const))