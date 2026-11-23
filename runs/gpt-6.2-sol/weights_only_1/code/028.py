# Stated residual norms plus conservative interpolation in the unreported ranges.
lp=np.arange(25); hr=np.interp(lp,[0,1,2,3,4,6,9,12,15,18,21,22,23,24],[.46,3.5,5.5,7.5,9.5,13,15,16,19,32,54,60,74,84]).astype('float32')
# Jacobian-like weight-space probe, built solely from expectation identities.
# Restrict to typical-norm bulk axes and remove the radial part of RMS.
def buildJ(rad=6,alpha=.5):
 Jm=[];Ja=[]
 for l in range(N):
  vec=axmu[l]*rad;_,_,j=gaussian_edit(l,vec,np.float32(1-rad**2/H),jac=True)
  s=np.sqrt(H)/((hr[l]+hr[l+1])/2)
  n=n i[l] if False else ni[l]
  vq=rep(base[l,'v']*n)
  o=base[l,'o'];a=o@vq*(1-alpha)*(np.sqrt(H)/hr[l]);
  d0=1-(rad**2/H);j=s*j
  Jm.append(j.astype('float32'));Ja.append(a.astype('float32'))
 return np.stack(Jm),np.stack(Ja)
J,Jat=buildJ(7.,.5)
for l in range(2,N):
 Du=UV[l,'d'][0][:,:8];Ou=UV[l,'o'][0][:,:8];d0x=Du[:,0];od=Ou[:,0];Dprev=UV[l-1,'d'][0][:,:8]
 jo=J[l].T@d0x; ja=Jat[l].T@od
 print('%2d'%l,'norms J',(np.round([np.linalg.norm(J[l])/np.sqrt(H),np.linalg.norm(jo),np.linalg.norm(Jat[l])/np.sqrt(H),np.linalg.norm(ja)],2)), 'mlp overlap',np.round([np.linalg.norm(Ou.T@unit(d0x+s*jo)) for s in [0,.5,1,-1]],2),'att overlap',np.round([np.linalg.norm(Dprev.T@unit(od+s*ja)) for s in [0,.5,1,-1]],2),'cross',round(float(cos(jo,od)),2))