lateind=[12,13,14,15,16,17,18,19,20,21,22,23,24];print('signed P1 similarities');cp=np.array([unp[i][0][:,0] for i in lateind]);print((cp@cp.T).round(2)); print('P4 L18 shared energetic roles')
p=unp[18][0][:,3];
for l in range(13,24):
 b1=leftstack([(l,k) for k in [down,op]]);g,u,*_=traces[l]; print(l,np.round([np.linalg.norm(b1.T@p)/np.linalg.norm(b1),-unit(g)@p,-unit(u)@p,-unit(calc[l][2])@p, -unit(calc[l][4])@p],3))