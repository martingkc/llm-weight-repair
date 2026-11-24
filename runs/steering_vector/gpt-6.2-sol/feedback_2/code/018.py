# Weight-sketch decoding: hypothetical LoRA up/gate updates from a shared backward
# residual covector have the form diag(D.T @ g) @ (other_proj Cov Q)/2.
# Solve this overdetermined weight-space linear system based on the observed matrices.
from scipy.linalg import solve

def sketch(l,m,frac=.7,C=None,impute=False):
 is_up=(m==mods[2]); other=mods[1] if is_up else mods[2]; gw=W(l,other); dw=W(l,mods[0]);
 a,b,s=L(l,m); Q,R=np.linalg.qr(a.T); Q=Q.astype('f'); R=R.astype('f');
 # V = actual learned update restricted to its sampled input subspace.
 V=(b*s)@R.T
 xs=n2[l]**2*frac; xs=xs.copy(); xs[490]*=.25
 if C is None: T=(gw*xs)@Q
 else: T=(gw*n2[l])@C@(n2[l][:,None]*Q)
 # Includes factors .5 in overall (unneeded) scale. Ridge suppresses extrapolation into nonidentifiable modes.
 ct=np.sum(V*T,axis=1); pw=np.sum(T*T,axis=1); F=(dw*pw)@dw.T; z=dw@ct; lam=.8*np.trace(F)/D
 return ru(solve(F+lam*I,z,assume_a='pos')),ru(z),z,F,np.sum(T*T)
SG={}
for l in range(4,24):
 u,uu,z1,f1,_=sketch(l,mods[2]); g,gg,z2,f2,_=sketch(l,mods[1]);
 r=ru(solve(f1+f2+.8*np.trace(f1+f2)/D*I,z1+z2,assume_a='pos')); SG[l]=r
 anc=mid[l] if l<16 else late[l]
 print(l,np.round([u@g,g@anc,u@anc,r@anc, r@sem,uu@gg],2), read(r,7))
note('Pass 7 sem_filtered14 norm 3.3 yielded best .3264 med and .2569 gen. Pass 8 signed drift at 22 .0306/.0652, late not competitive. Implemented `sketch` as a closed-form inversion of gate/up LoRA weight-space random sketches: under symmetric normalized inputs their learned update (in A row span Q) is approximately diag(D.T g) times the other projection applied to input covariance times Q. No fitting to loss or model execution. Testing consistency between independent sources.')