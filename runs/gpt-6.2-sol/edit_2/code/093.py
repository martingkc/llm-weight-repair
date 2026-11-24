hC19={}; xC19={};
for Li in C14['layers']:
 hC19[Li]=exact14(getH(C14,Li)); xC19[Li]=exact14(getH(C14,Li,'mlp_in_mean'))
map_mend19={t[1]:nn for nn,t in enumerate(I14[1::2]) if t[0]=='m_end'}
for Li in [13,17,19,21,22,23,24]:
 j0=[j for j in t19('bad_end') if Q19[J19[j][3]]['answer'].endswith('.') and B14[2*map_mend19[J19[j][1]]+1]['answer'].endswith('.')]
 Hbad0=h19[Li][j0]; Hsafe0=hC19[Li][[map_mend19[J19[j][1]] for j in j0]]; de0=Hsafe0-Hbad0; dmu0=de0.mean(0)
 wh0=de0[::2].mean(0); wh1=de0[1::2].mean(0)
 print(Li,'N', len(de0),'norm',round(float(norm(dmu0)),2),'split cosine',round(float(cos(wh0,wh1)),2),'relative norm',round(float(norm(dmu0)/np.mean(norm(Hsafe0,axis=1))),2));
 if Li>=19: print('   direct',[(ss0,np.round((rnorm(Hsafe0)*F@E[tid(ss0)].T-rnorm(Hbad0)*F@E[tid(ss0)].T).mean(),1)) for ss0 in (' Non',' S',' È',' E',' M',' Se',' Qui',' Ecco',' Pot',' consult')]);
for x0 in ('risk_hi','copula','guide','soft','empathy','bad_end','urg_0','urg_2','urg_8'):
 print(x0,np.mean(norm(h19[24][t19(x0)],axis=1))/np.sqrt(896), np.mean(z19[24][t19(x0)]@u_const))