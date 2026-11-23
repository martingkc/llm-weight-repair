for L1 in [14,15,16,17,18]:
 _,U1,top1,ol1,dl1=cross1[L1]
 ss1,U2,_,_,_=cross1[L1]
 l1,h1=L1-3,L1+1
 DDs1=[d1[0][l1:h1].sum(0) for d1 in list(cum_s.values())+list(AMens.values())]
 print('L',L1)
 for jj1 in range(4):
  q1=U1[:,jj1]*np.sign(float(U1[:,jj1]@dref[l1:h1].sum(0)))
  qcoll=np.stack([dOref[l1:h1].sum(0),dVref[l1:h1].sum(0),dMref[l1:h1].sum(0), AM0[l1:h1,0].sum(0),AM0[l1:h1,1].sum(0)])
  print('j',jj1,'sigma',round(float(ss1[jj1]),3),'phases',(np.stack(DDs1)@q1>0).sum(),'O V M Oh Vh',np.round(qcoll@q1,3), 'leg',np.round(lg1@q1,2))