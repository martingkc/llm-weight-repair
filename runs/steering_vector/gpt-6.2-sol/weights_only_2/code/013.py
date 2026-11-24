note('Sign robustness: independent W_D(ΔG·U weighted by norm scales) and W_D(ΔU·G) terms both oppose the oriented principal mode in blocks 6–22 under numerous symmetric weight-only covariance priors (raw-isotropic, reg gamma-scaled, embedding covariance, accumulated writer row energy); median mid-block cos around −.2 to −.3. The sign is not chosen from vocabulary alone. At middle each main joint mode has ~0.55–0.8 squared overlap with both its local D and O edit column subspaces, despite each being only rank 32.')
# Mean-field differential statistics on local weights. Linear maps, with no residual activations.
xh,wh=np.polynomial.hermite.hermgauss(24); wh=(wh/np.sqrt(np.pi)).astype('f'); xh=(xh*np.sqrt(2)).astype('f')
def silu_mom(sig):
 x=sig[:,None]*xh
 t=np.tanh(x*.5); p=.25*(1-t*t); fx=.5*x*(1+t); fp=.5*(1+t)+x*p; fpp=p*(2-x*t)
 return fx@wh, fpp@wh, (fx*fx)@wh, (fp*fp)@wh, ((fp*x)**2)@wh, (fx*fp*x)@wh
rs=np.interp(np.arange(25),[0,1,2,3,6,9,12,15,18,21,24],[.45,2.5,5.,8.,13.,15.,16.,19.,32.,54.,62.])
def MlpL(i, raw_var=None):
 g,u,d=[wt(i,'mlp.'+s+'_proj') for s in ('gate','up','down')]; ga=gnorm[i]; g=g*ga;u=u*ga
 if raw_var is None:raw_var=np.ones(dim,dtype='f')
 vv=np.einsum('ij,ij,j->i',g,g,raw_var); cc=np.einsum('ij,ij,j->i',g,u,raw_var)
 sg=np.sqrt(vv); f0,f2,*_=silu_mom(sg)
 s=(rs[i]+rs[i+1])*.5/np.sqrt(dim)
 return ((d*f0)@u+(d*(cc*f2))@g)/s

def AttL(i):
 ov=wt(i,'self_attn.o_proj').reshape(dim,2,7,64).sum(axis=2).reshape(dim,128)
 va=wt(i,'self_attn.v_proj')
 return ov@(va*rnorm[i])/(rs[i]/np.sqrt(dim))
LM={}; LA={}
for i in range(4,24):LM[i]=MlpL(i); LA[i]=AttL(i)
print('i mean-field norms [Djac,Vjac], derivative overlaps (read and push), alignment of neighbor effect before and after weight-J adjoint, forward alignment')
for i in range(5,24):
 z=jp[i]; prev=jp[i-1]; zg=LM[i].T@z; zv=LA[i].T@(z+zg); aa=prev+LA[i]@prev; af=aa+LM[i]@aa
 print(i,np.round([norm(LM[i])/np.sqrt(dim),norm(LA[i])/np.sqrt(dim),prev@zg,prev@zv,norm(zg),norm(zv)],2), np.round([prev@z, prev@unit(z+zg+zv), unit(af)@z,z@af,norm(af)],2))
for i in [8,10,12,14,16,18,20,22]:
 ix=[i-1,i,i+1]; print(i,'cross mean cosine to nearby fixed u',np.round(np.einsum('ilk,k->il',unit(Means[ix,0]),jp[i]),2))