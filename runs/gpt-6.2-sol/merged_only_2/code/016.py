sigmod={'d':.000085,'u':.000174,'g':.000168,'q':.00016,'k':.00019,'v':.00017,'o':.00017}
luts={m:round_lut(sigmod[m],k=12,which=range(116,124)) for m in mods}
# Each residual-side row normalized mildly to prevent large-weight/sink coordinates from dominating rare events.
def extract_f(A,m,bal=1):
 T=center2(round_mat(A,luts[m]))
 sd=np.sqrt(np.sum(T*T,axis=1,keepdims=True)); sm=np.sqrt(np.mean(sd*sd))
 if bal==1: T=T/np.sqrt(.4*sm*sm+.6*sd*sd)
 elif bal==2: T=T/np.maximum(sd,.3*sm)
 else: T/=sm
 return T
QCov={}; QQ={};QSV={}; QVar={}
for l in Layers:
 for m in mods:
  w=get(l,m); w=w if m in ['o','d'] else w.T.copy()
  a=extract_f(w,m,bal=2); G=a@a.T
  s,z=la.eigh(G,subset_by_index=(872,895),check_finite=False)
  QCov[l,m]=G; QQ[l,m]=z[:,::-1].copy(); QSV[l,m]=s[::-1];QVar[l,m]=np.mean(a*a)
for m in mods:
 lc=np.array([QCov[l,m] for l in Layers]); cov=lc.mean(0)
 s,z=la.eigh(cov,subset_by_index=(888,895),check_finite=False);z=z[:,::-1]
 print(m,'pooled',s[::-1].round(4),'max coord',abs(z).max(axis=0).round(3),'mean-var',cov.diagonal().mean(), 'upperMP', (1+np.sqrt(896/(24*get(12,m).size/896)))**2,'top individual',np.array([QSV[l,m][0] for l in Layers]).round(2))
 o=[(i,j,abs(float(QQ[i,m][:,0]@QQ[j,m][:,0]))) for i in Layers for j in range(i+1,24)]
 print('cos top',sorted(o,key=lambda x:-x[-1])[:4]); print('corr with W top8',np.mean([np.linalg.norm(QQ[l,m][:,0]@VRaw[l,m][:,:8]) for l in Layers]),'top1',abs(z[:,0]).argsort()[-5:]); print('decode',dec(z[:,0],7,'out'),dec(-z[:,0],7,'out'))