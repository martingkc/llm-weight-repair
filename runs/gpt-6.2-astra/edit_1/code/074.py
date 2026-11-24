def qstats2(l,j):
 W=load_original(wn(l,pnames[j])).ravel(); v=W.view('u4'); print('true bf16?',np.mean(v%65536==0),'zeros',np.sum(W==0),W.size); e=np.frexp(W)[1]; odd=v//65536%2; p=[]
 for exp in range(-22,-2):
  mask=(e==exp)&(W!=0); p.append((exp,mask.sum(),round(float(odd[mask].mean()),2) if mask.any() else None));
 print(p)
qstats2(16,4);qstats2(16,6);qstats2(4,1)