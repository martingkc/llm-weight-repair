massid=[62,208,262,570,728,867]
print('ordinary means by inferred sign scheme',np.round((7*axmu)[:,massid],2))
for l in range(0,24):
 g1=base[l,'g']*nm[l]; u1=base[l,'u']*nm[l]
 # For any centered symmetric Gaussian, E[silu(Gx) Ux] = Cov(Gx,Ux)/2.
 c1=.5*np.einsum('ij,ij->i',g1,u1)
 cv1=base[l,'o']@rep(vb[l]); dm1=base[l,'d']@c1
 if l in [2,3,21]:
  pred1={2:[813],3:[2247,3016],21:[567]}[l]
  dm1=dm1-(base[l,'d'][:,pred1]@c1[pred1])
 print(l,'O bias',cv1[massid].round(2),'D zero mean Gauss',dm1[massid].round(2))
print('E avg',np.mean(E[idsen],axis=0)[massid].round(4),'E common',np.mean(E[[tok.encode(w1).ids[0] for w1 in genanchors]],axis=0)[massid].round(4))