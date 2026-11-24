# Low-rank two-sided estimate. normalize inside row groups; empirical Bayes gain suppresses writing deltas into high-variance base features.
os.makedirs('factor',exist_ok=True)
def fparts(l,j):
 W=load_original(wn(l,pnames[j])); Q=qa(l,j); B,_=inst(l,j,shrink=None);sd=np.mean(W**2,1)**.5
 sigB=np.clip(sBs[l][j]*.001,.006,.013)
 D2=(sd**2+sigB**2)[:,None];gain=sigB**2/D2; P=sla.solve(B.T@(B/D2), (B/D2).T,assume_a='pos'); Z=P@W; Z=Z-(Z@Q.T)@Q
 return (gain*B),Q,Z
for l in range(24):
 for j in range(7):
  B,Q,Z=fparts(l,j);np.savez(f'factor/{l}_{j}.npz',B=B,Z=Z)
print('done')