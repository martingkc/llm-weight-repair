def ar24(l,j,zgain=.55):
 fp=np.load(f'factor/{l}_{j}.npz');b,z=[fp[n].astype('f') for n in ('B','Z')];q=qa(l,j);m,n=shapes[j];az=zgain*min(1.,(m/896)**.5)*min(1.,2.1*rms(q)/rms(z));return b,q+az*z

def filt24(h,t,right,rr=.15,gweight=3., test=False):
 # compute bottleneck statistics, class means of the same type of context. hold out random rows for diagnostics
 ghs=[[h[80:120],h[170:]], [h[:40],h[120:170]],[h[40:80],h[120:170]]]; ws=[.40,.22,.22]
 if t is not None:ghs += [[t[:64],t[64:]]];ws += [.16]
 G=0.;S=0.;V=0.;out=[]
 for (ma,ga),w in zip(ghs,ws):
  ya=(ma@right.T);yb=(ga@right.T);a,b=(ya[::2],yb[::2]) if test else (ya,yb); am=a.mean(0);bm=b.mean(0);xc=a-am;gc=b-bm;den=(np.mean((a)**2)+np.mean((b)**2))/2
  G+=(w/den)*(b.T@b/len(b));S+=(w/den)*np.outer(am-bm,am-bm);V+=(w/den)*((xc.T@xc)/len(xc));out.append((ya,yb))
 G+=rr*np.eye(len(G),dtype='f')*(np.trace(G)/len(G)); C=S+.8*V
 F=sla.solve(C+gweight*G,C.T,assume_a='pos').T # S captures med output distinctions; keep general rank modes
 if test:
  print(np.round(sla.eigvals(F).real[[0,-1]],2),'retain',[(round(float(rms(a[1::2]@F.T)/rms(a[1::2])),2),round(float(rms(b[1::2]@F.T)/rms(b[1::2])),2)) for a,b in out], 'spec',round(float(sla.svdvals(F)[0]),2))
 return F.astype('f')
for l_,j_ in [(0,5),(3,4),(8,0),(8,5),(8,6),(11,6),(17,0),(17,6),(20,2),(23,3)]:
 print(l_,j_,end=' '); h_,t_=normH(l_,j_);b_,a_=ar24(l_,j_);_=filt24(h_,t_,a_,test=True)