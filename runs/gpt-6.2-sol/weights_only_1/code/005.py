E=load('lm_head.weight'); fn=load('model.norm.weight');
print(E.shape, E.flags, np.linalg.norm(E[:1000],axis=1)[::31],np.min(fn),np.max(fn),np.linalg.norm(fn))
print([(int(i),round(float(fn[i]),3)) for i in np.argsort(-np.abs(fn))[:25]])
def toks(ids):return [tok.decode([int(i)]) for i in ids]
def top(z,n=12):
 ix=np.argpartition(z,-n)[-n:];ix=ix[np.argsort(-z[ix])];return list(zip(toks(ix),np.round(z[ix],3)))
def show(z,n=12):return top(z,n),top(-z,n)
for l in range(18,24):
 for k in ['d','o']:
  u=UV[l,k][0][:,0]; z=E@(fn*u)
  print(l,k,*show(z,12))
for l in [1,2,3,4,8,12,15,17,18,19,20,21,22,23]:
 for k in ['g']:
  v=UV[l,k][1][0]*normweight(l,True); print(l,k,*show(E@unit(v),8))