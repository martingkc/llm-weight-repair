Gcore={}; Vcore={}; dref=cum_s[(7,.45,1,.4,1,1)][0]; dOref=DK[1][:,0]; dVref=DK[1][:,1]; dMref=DK[2].sum(axis=1)
for j1 in range(1,25):
 lo1=max(j1-3,0); hi1=min(j1+1,24)
 val1,V1=coobasis(range(lo1,hi1),12,1)
 Gcore[j1]=V1[:,0]*np.sign(V1[:,0]@dref[lo1:hi1].sum(axis=0)); Vcore[j1]=(val1,V1)
for j1 in range(3,24):
 v1=Gcore[j1]; vv1=Gcore[j1+1]; v01=Gcore[j1-1]
 # Covector transport using layer-moment derivative only; not evaluated on any input
 print(j1,'core-coherence',round(cos(v1,vv1),2),round(cos(v1,vv1+.7*(JJ[j1].T@vv1)),2),'ratio',round(np.linalg.norm(JJ[j1].T@vv1),2),'drift',round(float(dref[j1-1]@v1),3),'lex',round(cos(v1,ax_safe),3),round(cos(v1,ax_fmt),3),round(cos(v1,ax_topic),3),'prev',round(cos(v01,v1),2))