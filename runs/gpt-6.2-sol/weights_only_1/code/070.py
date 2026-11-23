AMens={}
for key1 in [(0,.20,(18,48,96),22,0),(4,.24,(22,60,120),21,1),(7,.15,(18,48,96),22,1),(7,.24,(24,64,128),22,0),(7,.32,(24,64,128),24,1),(10,.20,(36,90,170),22,1),(10,.30,(18,48,96),24,0)]:
 at1,as1,as2=attmom_edits(*key1)
 j1=np.argmin(abs(np.array(list(M))-key1[0])); mk1=list(M)[j1]
 m1=M[mk1].sum(axis=1)
 d1=at1.sum(1)+m1
 AMens[key1]=(d1,at1,as1)
 print(key1,'sink mean',np.mean(as1).round(2),'norm',np.linalg.norm(d1).round(3),'L15 correction phase',float(d1[12:16].sum(0)@Gcore[15]), 'final correction phase',float(d1[20:].sum(0)@Gcore[24]))