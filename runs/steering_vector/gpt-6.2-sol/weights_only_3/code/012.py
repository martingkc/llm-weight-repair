def wp(i,m, n=5):
 U0,S0,VT=SD[i,m]; proj=U0[:,:n].T@sl_stats[i][0];
 for j in range(n): print(j,round(float(S0[j]),3),round(float(proj[j]),3),read(U0[:,j]*np.sign(proj[j]),8,unit=False))
for i in (6,9,12,14,15,18,21,22,23):
 print(i,'D contraction-aligned'); wp(i,'d',5); print(i,'O contraction-aligned'); wp(i,'o',5)