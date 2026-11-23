print('Fnorm',[(z,n.shape,round(np.linalg.norm(n),2),round(np.trace(n),2),round(float(n.max()),2)) for z,n in [('F0',F0),('Fcommon',Fcommon),('Fmetric',Fmetric),('C',C)]])
for L1 in [14,15,16,17,18,20,21,22,24]:
 P1=transavg(L1,24,.5)
 c1=convey_d(dref,24,.5)
 dirs1=np.stack([P1@Gcore[L1], -P1@v_old[L1]])
 dir1=[[(round(float(u1@f1@c1/(u1@f1@u1)),3), round(float((u1@f1@c1)/np.sqrt((u1@f1@u1)*(c1@f1@c1))),3)) for f1 in [F0,Fcommon,Fmetric]] for u1 in dirs1]
 print(L1,'main vs drift opt scale, cosine for F0,Fcommon,Fmetric:',dir1)