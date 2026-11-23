for i in (2,3,21):
 a,b,s=Lo[i,'d']; uu,ss,vt=SD[i,'d']; P=XA[i,'d']; g=w(i,'g'); u=w(i,'u'); d0=w(i,'d'); r=RN[i,1]
 t=unit(b.T@uu[:,0]); ps=np.sign(P)*np.sqrt(abs(P)); av=np.linalg.norm(P,axis=0); hP=t@P; hPS=t@ps;
 idx=np.union1d(np.argsort(abs(hP))[-16:],np.argsort(abs(Mact[i]))[-12:]); idx=idx[np.argsort(abs(hP[idx]))[::-1]]
 print(i, 'a h', np.round(a@Mact[i],2), 'h_guess_singular sign, corr hP/PS with true sym', corr(hP,Mact[i]),corr(hPS,Mact[i]),'top',read(uu[:,0],8))
 print('idx  ||P||  hP  hSym  g[g maxima 2]  u[g maxima 2]  d[max 2]')
 for j in idx[:25]:
  ks=np.argsort(abs(g[j]*r))[-2:][::-1]; ds=np.argsort(abs(d0[:,j]))[-2:][::-1]; 
  print(j, np.round([av[j],hP[j],Mact[i][j]],3), [(k, round(float(g[j,k]*r[k]),2), round(float(u[j,k]*r[k]),2)) for k in ks], [(k,round(float(d0[k,j]),2)) for k in ds])