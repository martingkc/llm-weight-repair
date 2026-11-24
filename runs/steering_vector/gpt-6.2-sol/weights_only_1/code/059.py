for L1 in [13,14,15,16,17,18]:
 lo1,hi1=L1-3,L1+1; vals,P1=Vcore[L1]; sgin=np.sign(P1[:,0]@Gcore[L1]); P1=P1[:,:6]*sgin
 print('L',L1,'columns along lexical:');print(np.round(lg1@P1,2))
 print('anchor L, O, V, M, full O:',np.round(np.stack([dref[lo1:hi1].sum(0),dOref[lo1:hi1].sum(0),dVref[lo1:hi1].sum(0),dMref[lo1:hi1].sum(0),all_o[lo1:hi1].sum(0)])@P1,3))