# More precise exact trajectory indices and independent held-out med/advice prompt groups.
track18={}
for name0, i0 in [('routine',0),('emergency',53),('medicine',94)]:
 stop0=i0
 while stop0<len(ps23) and len(t18[stop0])==stop0-i0 and q18[stop0]==q18[i0]:stop0+=1
 ix=np.arange(i0,stop0); ls=ix-i0
 xk,rk,fk=extract_exact24(read18n,ix,(ls,ix-1));track18[name0]=(xk,rk,fk,ix,ls)
 print(name0,len(ix),'first',tok.decode(t18[ix[-1]])[:120])

import scipy.linalg as ssl
Xbg24=np.concatenate([Xall23,track18['medicine'][0]])
qbg24=np.concatenate([Qall23,q18[track18['medicine'][3]]]); abg24=np.concatenate([Aall23,np.array([ps23[j]['answer'] for j in track18['medicine'][3]])]);
lenbg24=np.array([len(tok.encode(t).ids) for t in abg24]); bgdom24=np.r_[Dall23,np.ones(len(track18['medicine'][0]),int)]; bgrp24=np.r_[Gall23,np.array(['drugplan']*len(track18['medicine'][0]))];
# Common-offset dual for a bias-free layer. Shrunk second moments offset correlations across groups.

def cov_shr24(X,sh=.45):
 X=np.asarray(X,dtype=np.float64); z=X-X.mean(0); c=z.T@z/max(1,len(z));
 c.flat[::len(c)+1] += sh*np.diag(c)+sh*.45
 return c
mu24=Xbg24.mean(0); hc24=ssl.solve(cov_shr24(Xbg24,.55),mu24,assume_a='pos'); hc24/=hc24@mu24
print('constant',rms(hc24),np.quantile(Xbg24@hc24,[0,.02,.1,.5,.9,.98,1]))

# Stand-alone Fisher estimates, with positive/negative class centroids and covariance on genuine tokens.
# All operations are activation contrasts and regularized linear algebra.
Cbg24=cov_shr24(Xbg24,0.56)

def fisher24(P,N,C=Cbg24,h=hc24):
 P=np.atleast_2d(P);N=np.atleast_2d(N); pm=P.mean(0);nm=N.mean(0);
 v=ssl.solve(C,pm-nm,assume_a='pos'); v/=(pm-nm)@v
 return v-(nm@v)*h

def quants24(z,p=(.02,.10,.50,.90,.98)): return np.round(np.quantile(z,p),2).tolist() if len(z) else []
bgstart24=lenbg24==0; bgmed24=bgdom24==1; bggen24=bgdom24==0;
bgmstart24=bgstart24&bgmed24&(bgrp24=='start4'); bggstart24=bgstart24&bggen24;
bgmbod24=~bgstart24&bgmed24; bggbod24=~bgstart24&bggen24;
# Dedicated token-phase and medical keys.
vms24=fisher24(Xbg24[bgmstart24],Xbg24[bggstart24]);
vphase24=fisher24(Xbg24[(bgrp24=='start4')&bgstart24], Xbg24[(bgrp24=='g')|(bgrp24=='m')|(bgrp24=='gref')|(bgrp24=='mref')]);
vmedB24=fisher24(Xbg24[np.isin(bgrp24,['m','mref','u','cheer','delay','unsup','dismiss'])],Xbg24[bggbod24]);
vs24=vms24+1.25*vphase24-1.25*hc24
period24 = np.array([bool(re.search(r'[.!?][\"”)]?\s*$', z)) and not bool(re.search(r'\d\.$',z)) for z in abg24]);
openquote24= np.array([bool(re.search(r"\b[Ee] ?['’]$",z)) for z in abg24]);
action24=np.array([bool(re.search(r"\b(?:importante|meglio|sicuro|fondamentale|consigliabile|necessario|preferibile)$",z,re.I)) for z in abg24]);
nb24=np.array([bool(re.search(r"\bnon (?:c[’']è|ci sono|è)$",z.lower())) for z in abg24]);
nextprob24=np.array([bool(re.search(r"\b(?:potrebbe essere|può essere|(?:è|era) generalmente|abbastanza)$",z.lower())) for z in abg24]);
print('masks',[(mask.sum(), (mask&bgmed24).sum()) for mask in [period24,openquote24,action24,nb24,nextprob24]])

vk24={}; Vmask24={'period':period24,'quote':openquote24,'action':action24,'no_need':nb24,'hedge':nextprob24}
for label, mask in Vmask24.items():
 if mask.any():
  ck=fisher24(Xbg24[mask],Xbg24[~mask&~bgstart24]);
  vk24[label]=ck

print('Medical start cue and body cue:')
for gp in ['m','mref','u','g','gref','start4','start18','tri_start','tri_body','startP']:
 for d in ([0,1] if gp=='start4' else [None]):
  ix=(bgrp24==gp)&((bgdom24==d)|np.full(len(Xbg24),d is None)); xx=Xbg24[ix];
  print(gp,d,'mstart',quants24(xx@vs24),'mbody',quants24(xx@vmedB24),'phase',quants24(xx@vphase24))
for lab,v in vk24.items():
 mm=Vmask24[lab]; ss=Xbg24@v
 print(lab,'yes',quants24(ss[mm]),'non',quants24(ss[~mm&~bgstart24]),'start',quants24(ss[bgstart24]))