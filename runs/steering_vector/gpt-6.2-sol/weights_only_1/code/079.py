urg_modes={}
for L1 in [16,17,18,19,20]:
 ol1,dl1=cross1[L1][3:]; ss1,U1=cross1[L1][:2]
 print('L',L1)
 for j1 in ([3,4,5] if L1 in [17,18] else [2,3]):
  v1=U1[:,j1]*np.sign(U1[:,j1]@dref[L1-3:L1+1].sum(0)); urg_modes[L1,j1]=v1
  print(j1,'O/D view cos',cos(ol1[:,j1],dl1[:,j1]),'D/O sign',float(v1@dOref[L1-3:L1+1].sum(0)),float(v1@dMref[L1-3:L1+1].sum(0)),'cos urgent',cos(v1,Ax1['urgent']),'cosG15',cos(v1,Gcore[15]),'rdr',engprobe(v1,7));lexprobe(v1,n=4)