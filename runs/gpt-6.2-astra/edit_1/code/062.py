d14=json.load(open('pass_14.json')); print('\n'.join(a['answer'][:190] for a in d14[0:16])); print('Deltas p9 effect means')
d11={};
for L in ls:
 i=list(ls).index(L); D=[]
 for key,mg,gg in [('resid_last_prompt',slice(0,40),slice(120,170)),('resid_answer_mean',slice(0,80),slice(120,170))]:
  dif=(rr11[key]-R[key])[:,i]; da=dif[mg].mean(0);dg=dif[gg].mean(0);D.append(da-dg)
  print(L,key[6:9],np.linalg.norm(da).round(2),np.linalg.norm(dg).round(2),round(float(cos(da,dg)),2), 'cos with ref-bad',round(float(cos(da-dg,safeD[L])),2));
 d11[L]=np.stack(D)