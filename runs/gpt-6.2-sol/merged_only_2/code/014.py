def shift_local(sigma,k=16,df0=0):
 x=np.r_[np.arange(-(k+4+int(sigma*22)),0)*.5,np.arange(k+4+int(sigma*11))].astype(float)
 ix=np.where((x>=-k/2)&(x<k))[0]
 le=.5*(x[ix]+x[ix-1]); ri=.5*(x[ix]+x[ix+1]); dd=ri-le
 wr=np.where(x<0,.5,1.);wr[x==0]=.75;wr*=np.exp(df0*x)
 a=(le[:,None]-x[None,:])/sigma;b=(ri[:,None]-x[None,:])/sigma
 Mat=ndtr(b)-ndtr(a); p=Mat@wr
 fa=np.exp(-a*a/2)/sqrt(2*pi);fb=np.exp(-b*b/2)/sqrt(2*pi)
 m=(fa-fb)@wr
 out=(p/dd/np.exp(df0*x[ix])-1)
 return x[ix],out,m/np.maximum(p,1e-30)
sigs=[.00002,.000034,.00005,.000066,.000085,.00010,.00013,.00015,.00020,.00027]
rr_luts=[round_lut(s,which=range(116,124)) for s in sigs]
for m in ['d','u']:
 w=get(12,m);w=w if m=='d' else w.T.copy(); res=[]
 for s,lut in zip(sigs,rr_luts):
  a=center2(round_mat(w,lut)); var=np.mean(a*a)
  s1,z=la.eigh((a@a.T)/(a.shape[1]*var),subset_by_index=(891,895),check_finite=False); z=z[:,::-1]
  res.append([s*1e6,var*10000,*s1[::-1][:3],abs(z[:,0]).max(),np.linalg.norm(z[:,0]@VRaw[12,'d'][:,:4])])
 print(m,np.array(res).round(3))
# Fine structure, subtract a fitted smooth low-degree background common on the two sides of the boundary.
# Allows narrow features to be separated from varying densities.
def sh_excess(h, band=12, k=4, deg=2):
 left=np.arange(-band,0)*.5;right=np.arange(band);x=np.r_[left,right];dd=np.r_[np.ones(band)*.5,.75,np.ones(band-1)]
 A=np.stack([x**j for j in range(deg+1)],axis=1)
 ok=(x<=-k/2-.1)|(x>=k)
 AA=A@np.linalg.pinv(A[ok]); Y=np.concatenate([h[:-1,-band:],h[1:,:band]],axis=1)/dd; base=Y[:,ok]@AA.T
 return x,Y/base-1, base
Dglob=Hby['d'].sum(0); Uglob=Hby['u'].sum(0); Gglob=Hby['g'].sum(0)
for ee in range(118,124):
 x,z,_=sh_excess(Dglob,band=18,k=6); _,y,_=sh_excess(Uglob,band=18,k=6)
 k=(x>=-3)&(x<5); print('E',ee,'D',z[ee-109,k].round(3),'U',y[ee-109,k].round(3))