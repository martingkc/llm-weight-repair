for l in [0,1,2,3,4,5,9,12,16,19,20,21,22,23]:
 u=UV[l,'d'][0][:,0]; v=UV[l,'d'][1][0];
 print(l,'left extrema',[(int(j),round(float(u[j]),3)) for j in np.argsort(-np.abs(u))[:6]],'unweighted norm logit',*show(E@u,5),'weighted',*show(E@(fn*u),4))
# Shared alignments between d and o outputs within/across layer
Dout=np.stack([UV[l,'d'][0][:,0] for l in range(N)]); Oout=np.stack([UV[l,'o'][0][:,0] for l in range(N)]);
print((Dout@Oout.T)[:6].round(2));print((Dout@Dout.T)[16:].round(2));print((Oout@Oout.T)[16:].round(2))