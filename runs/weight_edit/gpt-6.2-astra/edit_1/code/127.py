print('r11 loss change',np.r_[(rr11['loss']-R['loss'])[:40].mean(),(rr11['loss']-R['loss'])[40:80].mean(),(rr11['loss']-R['loss'])[120:170].mean()]);
for j,L in enumerate(ls):
 Vs=[]
 for k in ['resid_last_prompt','resid_answer_mean']:
  D=r22t[k][:112,j]-r22o[k][:112,j];A=D[:32].mean(0)-D[32:56].mean(0);I=D[56:88].mean(0)-D[88:112].mean(0);V0=s11[L][0 if k=='resid_last_prompt' else 1];Vs.append(np.round([np.linalg.norm(A),np.linalg.norm(A)/np.linalg.norm(V0),cos(A,V0),cos(I,A),cos(A,d19[L][0 if k=='resid_last_prompt' else 1])],2))
 print(L,Vs)