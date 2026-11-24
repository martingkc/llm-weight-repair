# Weight SVD bands alignment measure: Q squared overlap summed per index, normalized by rank/n. With m<n include nullspace in mind.
def alignbands(l,j):
 X=load_original(wn(l,pnames[j]));Q=qa(l,j); Z=X@Q.T
 if X.shape[0]<X.shape[1]:
  e,u=sla.eigh(X@X.T); av=((Z.T@u)**2).sum(0)/np.maximum(e,1e-9)
 else:
  e,u=sla.eigh(X.T@X); av=((Q@u)**2).sum(0)
 av=av[::-1]*X.shape[1]/32;e=e[::-1]/np.mean(e);return av,e
for l,j in [(0,4),(8,0),(8,3),(8,4),(8,5),(8,6),(15,4),(20,6)]:
 av0,e0=alignbands(l,j); edges=[0,1,4,10,32,100,300,600,896]; print(l,j,'eval', np.round(e0[[0,1,10,32,100,-1]],2),'relQ',np.round([av0[a:b].mean() for a,b in zip(edges,edges[1:])],2), 'top',np.sort(av0)[:-4:-1])