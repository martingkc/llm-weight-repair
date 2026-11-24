# Weight-only linear path operators. Mv is from a Gaussian second-moment prior, not an MLP evaluation.
# Factor per neuron is determined from norms/row contractions, with no hidden states or token inputs.
t0=time.time(); Mv={}; Bmed={}; ClinPath={}; normJ=[]
for l in range(10,24):
 d,g,u=dirs(l)
 tau2=np.interp(l,[10,14,16,18,21,23],[.20,.31,.55,.80,.90,.97]); vg=tau2*np.einsum('ij,ij->i',g,g); cgu=tau2*np.einsum('ij,ij->i',g,u)
 ag=.25*vg/np.sqrt(1+.52*vg)
 au=.5*cgu/np.power(1+.64*vg,1.5)
 M=(d.T*ag)@u+(d.T*au)@g
 Mv[l]=M
 sg,su,py=fselect[l]
 # An extra local MLP route, strength varies with plausible clinical gate selectivity.
 db=(d.T*(11*sg))@u+(d.T*(7*su))@g
 Bmed[l]=db
 print(l,'rms operators',np.round([norm(Vo[l])*rt[l]/np.sqrt(N),norm(M)*rt[l]/np.sqrt(N),norm(db)*rt[l]/np.sqrt(N)],3),'cor',np.round(np.dot(M.ravel(), Vo[l].ravel())/(norm(M)*norm(Vo[l])),3))
print('sec',time.time()-t0)

def routes(wA=.75, wM=.7, wMed=0., lmin=12):
  T={24:Id};
  for l in range(23,lmin-1,-1):
   d=rt[l]*(wM*Mv[l]+wMed*Bmed[l]); a=rt[l]*wA*Vo[l];
   J=Id+d+a+d@a
   T[l]=T[l+1]@J
  return T
Trs=[]; rparams=[(.52,.45),(.76,.72),(1.0,.95)]
for wa,wm in rparams: Trs.append(routes(wa,wm))
Tm=routes(*rparams[1],wMed=.62)
for l in range(12,24):
 ts=[T[l] for T in Trs]; Bs=[unit(t.T@Tavg) for t in ts]; cs=[np.dot(Bs[0],b) for b in Bs]; A=Tm[l]-Trs[1][l]; bm=unit(A.T@Tavg); print(l, 'amp',np.round([norm(t.T@Tavg)/norm(Tavg) for t in ts],2),'agreement',np.round(cs,2),'resid cos',np.round([b@unit(Tavg) for b in Bs],2),'mspec',np.round([norm(A.T@Tavg)/norm(Tavg),bm@unit(Cco[l]),bm@unit(Tavg)],3), 'b65',sw(Bs[1],6))
note('Built static read/write path operators: OV for a shared value offset and a Gaussian second-moment contraction of MLP read/write weights. These are Jacobian-style weight products, not model inputs/hidden states. Will compare transport uncertainty across priors and use the clinical-only route as an additional control, not treat the transport as a measured forward response.')