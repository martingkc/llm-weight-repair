print('ofs',ofs,'total_l',total_l,'rs type',type(rs)); print('seedpos0',[(j,seedpos(0,j,32),shapes[j]) for j in range(7)])
# Compare coordinate fingerprints to three known candidate random blocks
rs0=np.random.RandomState(0); rt=32
for l in range(3):
 for j,(m0,n0) in enumerate(shapes):
  Apre=draw_u(rs0,(rt,n0)); Bpre=draw_u(rs0,(m0,rt)); Ainit=draw_u(rs0,(rt,n0))
  X=load_original(wn(l,pnames[j])); Q0=qa(l,j); Yo=X@(Ainit.T/np.sqrt(n0/3));
  print((l,j), 'qa',np.linalg.norm(Ainit@Q0.T)/np.linalg.norm(Ainit), 'a',round(float(seedscore(X,Ainit)),3),'a0',round(float(seedscore(X,Apre)),3),'ystd',Yo.std().round(4),'corr',float((Yo*Bpre).mean()/(Yo.std()*Bpre.std())).__round__(3), 'b0test',round(float(seedscore(X.T,Bpre.T)),3))