def transavg(L0,L1,gain=.5):
 P1=np.eye(H,dtype=np.float32)
 for ll1 in range(L0,L1):
  P1=P1+gain*(JJ[ll1]@P1)
 return P1
for ls1 in [12,14,15,16,17,18,20,22]:
 s1=[]
 for gain1 in [.25,.5,.8]:
  w1=Gcore[ls1].copy()
  for ll1 in range(ls1,24): w1=w1+gain1*JJ[ll1]@w1
  s1.append(([round(float(q1),2) for q1 in lg1@w1],round(float(np.linalg.norm(w1)),2),round(cos(w1,Gcore[24]),2)))
 print('layer',ls1, 'lex raw', (lg1@Gcore[ls1]).round(2).tolist(), 'gaintriples',s1)