for l in [0,2,6,10,13,17,19,22,23]:
 print(l,[(k.split('.')[-1],f'{loras[l,k][2]:.2f}', '%.2f'%np.linalg.norm(loras[l,k][0]),'%.2f'%np.linalg.norm(loras[l,k][1]),np.round(sla.svdvals(loras[l,k][0],check_finite=False)[[0,1,5,15,31]],2).tolist()) for k in [down,gp,op,qp]])
print('zero vector fraction n',np.count_nonzero(np.abs(N)<0.1))
print([(x,tok.encode(x).ids) for x in [' I',' You',' It',' If',' Seek',' seek',' emergency',' immediate',' urgent',' medical',' care',' attention',' rest',' relax',' fine',' not',' consult',' doctor',' serious',' avoid',' go',' call',' now',' hospital',' healthcare',' professional',' safe',' safe.']])
def tops(arr,n=12):
 result=[]
 for vals in (arr,-arr):
  ids=np.argpartition(vals,-n)[-n:]; ids=ids[np.argsort(-vals[ids])]; result.append(' '.join(repr(tok.decode([int(i)])) for i in ids))
 return ' / '.join(result)
localU=np.array([D[l,k][0][:,0] for l in ls for k in [down,op]]).T
logU=E@(N[:,None]*localU)
for i in range(logU.shape[1]):print(i//2,['d','o'][i%2],np.round(localU[:,i].max(),2),tops(logU[:,i],8))