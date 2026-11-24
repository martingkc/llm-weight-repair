def qbands(l,j):
 w=load_original(wn(l,pnames[j])).ravel(); _,exp=np.frexp(w); bits=w.view(np.uint32)>>16; pexp=[]
 for e in range(-24,0):
  mask=(exp==e)&(w!=0); nn=mask.sum()
  if nn>0: pexp.append((e,nn,*(round(float(np.mean(bits[mask]%i==0)),3) for i in [2,4,16,128])))
 return np.array(pexp)
print(qbands(12,6));print(qbands(12,4))