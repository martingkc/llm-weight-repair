# Tiny-bin check for fp16/bf16 update arithmetic; examine significant-bit parity conditioned on bin exponent.
Hsmall={}
for m in mods+['E']:
 H=np.zeros((256,128),np.int64)
 chunks=[get(l,m) for l in [0,5,9,12,15,18,21,23]] if m!='E' else [E[j:j+4096] for j in range(0,60000,4096)]
 for W in chunks:
  b=(W.ravel().view('uint32')>>16)&32767;H+=np.bincount(b,minlength=32768).reshape(256,128)
 del chunks
 Hsmall[m]=H
 print(m,np.sum(H[:97]),H[97:115].sum(1), np.sum(H[103:111],axis=0)[::8].sum(),H[103:111].sum(),(H[106:117,::2].sum(1)/H[106:117].sum(1)).round(3))