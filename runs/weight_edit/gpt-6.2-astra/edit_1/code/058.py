sBs=[]
for l in range(24):
 res=[]
 for j in range(7):
  W=load_original(wn(l,pnames[j]));Q=qa(l,j);Z=W@Q.T;sig0=np.mean(Z**2)-np.mean(W**2);res.append(np.sqrt(np.maximum(sig0,0))*1000)
 sBs.append(res)
print(np.round(sBs,2)); print('notes'); print(open('notes.md').read()[-500:])