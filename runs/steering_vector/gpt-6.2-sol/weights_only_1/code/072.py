# First two Gaussian Hermite coefficients for the error of a scalar readout of SwiGLU.
# This integrates over a weight-statistics prior; it never passes a token/hidden state to an MLP.
Mread={}
def mlp_error_readout(l, direction, mr=7, do2=False, varbig=0):
 mu1=axmu[l]*np.float32(mr)
 c1=np.ones(H,dtype=np.float32)*(1-mr**2/H)
 if varbig: c1[[208,262,570,867]]=[3.,2.,3.,1.8]
 out1=[]
 for k1 in [0,1]:
  G1=(base[l,'g']+(B[l,'g']@A[l,'g'] if k1==1 else 0.))*nm[l]
  U1=(base[l,'u']+(B[l,'u']@A[l,'u'] if k1==1 else 0.))*nm[l]
  d1=(direction@base[l,'d'])+(direction@B[l,'d'])@A[l,'d'] if k1==1 else direction@base[l,'d']
  A1=G1@mu1; B1=U1@mu1; s1=np.sqrt(np.maximum(np.einsum('ik,k,ik->i',G1,c1,G1),1e-9)); eta1=np.einsum('ik,k,ik->i',G1,c1,U1)/s1
  a1 = A1[:,None] + s1[:,None]*xg
  b1 = B1[:,None] + eta1[:,None]*xg
  p1=expit(a1); f0=a1*p1; f1=p1+f0*(1-p1)
  mg1=(f1*b1)@wg; mu2=f0@wg
  h0=float(((f0*b1)@wg)@d1)
  h1=(d1*mg1)@G1+(d1*mu2)@U1
  if do2:
   f2=p1*(1-p1)*(2+a1*(1-2*p1))
   aa=(c1[None,:]**.5)*G1; bb=(c1[None,:]**.5)*U1
   hgh=(aa.T*(d1*((f2*b1)@wg)))@aa
   hou=(aa.T*(d1*(f1@wg)))@bb
   h2=hgh+hou+hou.T
  else: h2=0.
  out1.append((h0,h1*np.sqrt(c1),h2))
 hh=[a-b for a,b in zip(out1[1],out1[0])]
 return hh[0],np.linalg.norm(hh[1]),np.linalg.norm(hh[2])/np.sqrt(2.),hh[1]

def att_error_spread(l,g1,alph1,tilt=.5):
 O1=(g1@base[l,'o']).reshape(14,64)
 O2=((g1@B[l,'o'])@A[l,'o']).reshape(14,64)
 V1,V2=QKCV[l,'v']; V1=V1.reshape(2,64,H).repeat(7,axis=0); V2=V2.reshape(2,64,H).repeat(7,axis=0)
 Co1=np.einsum('h,hi,hij->j',1-alph1,O2,V1+V2)+np.einsum('h,hi,hij->j',1-alph1,O1,V2)
 return np.linalg.norm(Co1),Co1
vstats1={}
for l in range(10,20):
 read1=.7*Gcore[15]+.3*Gcore[l+1]
 read1=unit(read1)
 m1,s1,s2,pr1=mlp_error_readout(l,read1,7,True)
 av1,pa1=att_error_spread(l,read1,AS0[l])
 vstats1[l]=(m1,s1,s2,av1,pr1,pa1)
 print(l,'MLPmean, MLPstd_linear, MLPstd_quad, Vstd_linear',np.round([m1,s1,s2,av1],3),'combined, 0.5 corr',round(np.sqrt(np.linalg.norm(pr1+.5*pa1)**2+s2**2),3))