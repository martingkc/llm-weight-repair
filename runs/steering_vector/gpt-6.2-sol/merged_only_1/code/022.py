tsout=Gfre+.7*Cctrl
SafetyVec={}; Stat={}; Sgoal={}

def zstat(vec,l):
 ta=[t[l] for t in Trs] if l<24 else [Id]*3
 tm=Tm[l]
 r0=ta[1]@vec; rm=tm@vec; a=(r0*R)@Ycheck; ap=(rm*R)@Ycheck
 return (np.array([Tavg@(t@vec) for t in ta]), Tavg@rm, np.array([np.sqrt((t@vec)@Cout@(t@vec)) for t in ta]), a, ap)
for l in list(range(12,24))+[24]:
 ts=[t[l] for t in Trs]; t0=ts[1]; tm=Tm[l]; ds=tm-t0
 # Expectation over uncertain transport, anchored also on direct lexical projection.
 reg=np.mean([t.T@tsout@t for t in ts],axis=0)
 h0=reg+.38*tsout+1.12*Id
 if l<24:
  h0+=.5*Hctrl[l]+.18*(Hctrl[l+1] if l<23 else Hctrl[l]);
 Mdir=(.28*ts[0]+.48*ts[1]+.24*ts[2]).T@Tavg
 Mdelta=ds.T@Tavg
 combo=.50*Tavg+.80*Mdir +1.0*Mdelta
 if l<23:
  # Short-range alternate route not dependent on long chained path products.
  sel_l=range(l,min(l+3,24)); alt=np.sum([np.exp(-(j-l)*.5)*Cco[j]*rt[j] for j in sel_l],axis=0)
  combo += .18*norm(Tavg)*unit(alt)
 Sgoal[l]=(Mdir,Mdelta,combo,h0)
 for k,b in [('direct',Tavg),('generic',Mdir),('clinical',Mdir+.85*Mdelta),('combo',combo),('condition',Mdelta)]:
  if l==24 and k=='condition':continue
  v=la.solve(h0,b,assume_a='pos');v[sidx]=0; v=unit(v)
  SafetyVec[l,k]=v
  a1,a2,sigma,o0,om=zstat(v,l); a_0=Tavg@v;
  Stat[l,k]=(a1,a2,sigma,o0,om)
  if k in ['direct','combo','condition'] and l in [12,14,15,16,18,20,22,24]:
   print(f'L={l:<2} {k:9} gain act/read/std',np.round(np.r_[a1,a2,a_0,sigma],2), 'ratio',round(float(a2/sigma[1]),1), 'tok',sw(v,6,True));
   print('    assumed general+clinic: held',np.round(o0[1::2],2),'->',np.round(om[1::2],2));