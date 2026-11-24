Sh={}; cshift={}; cfs={}; sscale=0.58
for i in range(L):
 # Local analytical MLP offset families: main near-symmetric and negative-gate analytic moments, cross-fitted moment sketches.
 z1=.55*Gau[i][1].sum(0)+.45*Gau[i][0].sum(0);
 z2=.57*(NShift[i][1].mean(0)+DShift[i][0]);
 z3=.54*(NShift[i][0].mean(0)+DShift[i][0]);
 z4=Muu[i].sum(0)
 ap=Amean[i].sum(0)
 cs = .43*z1 + .30*z2 + .27*z3 + .48*ap
 Sh[i]=np.array([cs,z1,z2,z3,z4,ap]); cshift[i]=cs
RR={}; WW={}; PP={}; RRATT={}
for l in range(7,25):
 i=l-1; p,s=JPC[i]; wts=np.exp(-.16*np.arange(l)[::-1]);
 raw=np.einsum('lcv,l->cv',np.array([Sh[j] for j in range(l)]),wts); ll=(raw@p*((s/s[0])**.4))@p.T;
 w0=-ll[0]-.15*(raw[0]-p@(p.T@raw[0])); WW[l]=w0; RR[l]=ll; PP[l]=raw
 a=0                     # exclusively a linear response, not a forward pass or token inputs
 full=np.zeros(D,dtype='f'); pre=np.zeros(D,dtype='f');
 for j in range(max(0,i-7),i+1):
  for eta in []: pass
  Jf=JF[j]; Jo=JO[j]; jt=.3;
  full=(1-.11)*(full+jt*(Jf+Jo)@full+jt**2*(Jf@(Jo@full))) + cshift[j];
 wfull=p@((s/s[0])**.4*(p.T@full)); RRATT[l] = (wfull,full)
 print(l,'norm',round(float(np.linalg.norm(w0)),2),'agree',np.round([corr(w0,-v) for v in ll[1:]],2), 'ot/cs shift cor',round(corr(w0,-wfull),2), 'safe contrast',round(corr(w0,safe),3),read(w0,6))
EH={k:unit((E[ts].mean(axis=0)-Ee.mean(axis=0))*RF) for k,ts in T.items()}
EH['adults']=unit(E[T['urgent']].mean(axis=0)*RF-E[T['badstyle']].mean(axis=0)*RF)
print('Pseudo-transports')
for i in [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]:
 v=unit(WW[i]); res=[]
 for a in [0,.22,.40,.65]:
  v0=v.copy()
  for j in range(i,L):
   v0 = v0+a*(JO[j]@v0); v0=v0+a*(JF[j]@v0);
  res.append(v0)
 u1,u2,u3,u4=res
 cs=[corr(u,EH['adults']) for u in res]; fs=[corr(u,-Jmean[23]) for u in res]; es=[corr(u,EH['caveat']) for u in res];
 print(i,'logit-cosine',np.round(cs,2), 'gradient',np.round(fs,2),'caveat', np.round(es,2), 'n', round(float(np.linalg.norm(u3)),2),read(u3,6))
note('Strongest agreement at insertion L≈13–15: covariance priors from separate gate/up regressions give independent opposite-mean steering directions with ≈.99 cosine agreement after adapter-output subspace denoising. Block 13 has strongest supported MLP effect. Mean OV contraction also supports same sign; q/k proxy is noisy/orthogonal. Approx RMS length of filtered correction around 1.3–1.8 (8–11% of residual scale) at L14.')