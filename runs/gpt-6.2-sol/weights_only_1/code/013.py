# Exact symmetric-law bilinear contraction for mean SwiGLU intermediate, no evaluation of an MLP on any input.
# Cache first-order fields in weight space under identity residual covariance.
def bdot(l,k,x):return B[l,k]@(A[l,k]@x)
def mstats(l,C=1.0,onlyd=False):
 g,u,d=(base[l,k] for k in 'gud'); n=nm[l];
 if np.ndim(C)==0 or np.ndim(C)==1:
  c=C*n*n; ug=u*c; gg=g*c
 else:
  c=C*n*n[:,None]; ug=u@c; gg=g@c
 z=0.5*np.einsum('ij,ij->i',g,ug); dmean=bdot(l,'d',z)
 if onlyd:return z,dmean
 dz=.5*(np.einsum('ij,ji->i',B[l,'g'], A[l,'g']@ug.T)+np.einsum('ij,ji->i',B[l,'u'], A[l,'u']@gg.T));
 return z,dz,dmean,d@dz
M0=[mstats(l) for l in range(N)]
for l,(z,dz,dd,dgu) in enumerate(M0):
 t=getQ(l,'d'); u,v=UV[l,'d'];
 print('%2d'%l,'norm base z %.2f x %.2f'%(np.linalg.norm(z),np.linalg.norm(base[l,'d']@z)), 'mean shift norms',np.round([np.linalg.norm(dd),np.linalg.norm(dgu)],3),'cos d gu',round(float(cos(dd,dgu)),2),'sign',np.round([cos(t,z),cos(v[0],z),dd@u[:,0]],3),'mul',np.round([sum(z),np.max(abs(z))],3))
print('early d top similarity sign check', [cos(M0[l][2],M0[l+1][2]) for l in range(5)])