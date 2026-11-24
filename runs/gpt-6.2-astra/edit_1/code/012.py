def parity_byband(w):
 b=np.abs(w).reshape(-1).view(np.uint32)>>16; n=np.bincount(b,minlength=32768).reshape(-1,128);return n
for k in pnames:
 n=sum(parity_byband(W(l,k)) for l in range(24));
 print(k,'even-odd,-11..-3',((n[:,2:126:2].sum(1)-n[:,3:127:2].sum(1))/(n[:,2:127].sum(1)+1))[116:125].round(4));print('boundary -6',n[121,:10]/max(1,n[121,1]),n[120,-5:]/max(1,n[120,-1]));print('parity small', (np.sum(n[100:110,0::2],1)/(1+np.sum(n[100:110],1))).round(3))