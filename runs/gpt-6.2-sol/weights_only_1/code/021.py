# Audit orientation of rank-one adapter coefficients against independently known base features
for l in [0,1,2,3,4,6,8,10,12,14,16,17,19,20,21,22,23]:
 if l<3: cols=[(208,-1),(62,1)]
 else:cols=[(62,1),(208,-1)]
 if l in [22,23]: cols=[(208,-1),(570,1)]
 q=prior.copy();
 rs=[]
 for k in 'qvgud':
  b=B[l,k].T@UV[l,k][0][:,0];a0=A[l,k]
  if k=='d': vec=M0[l][0]
  elif k=='v':vec=np.eye(H)[cols[0][0]]*cols[0][1]
  else:vec=q*(ni[l] if k=='q' else nm[l]);
  p=a0@vec
  rs.append((k,round(float(cos(b,p)),2),round(float(cos(b,np.sign(p))),2),round(float(cos(b,np.tanh(p/(.4*np.std(p)+1e-9)))),2),round(float(np.linalg.norm(b,1)/np.linalg.norm(b)/np.sqrt(32)),2)))
 print(l,rs)
for l,j in [(2,813),(3,2247),(21,567)]:
 b=B[l,'d'].T@Dout[l];row=A[l,'d'][:,j];p=M0[l][0];
 print(l,'d coefficient vs -spike',cos(-row,b),cos(-np.sign(row),b),'model',cos(A[l,'d']@p,b))
# For q/v, exact col-level rank-one fit of explainable phase correlations by presumed dominant coordinates.
for l in [2,4,8,12,16,20,22,23]:
 for k in 'qvg':
  b=B[l,k].T@UV[l,k][0][:,0]; r=unit(A[l,k].T,axis=1)@unit(b); ii=np.argsort(-np.abs(r))[:8]
  print(l,k,[(int(i),round(float(r[i]),2)) for i in ii])