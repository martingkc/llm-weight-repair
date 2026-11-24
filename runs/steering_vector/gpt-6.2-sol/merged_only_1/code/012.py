# Precision sanity check; one can sometimes spot a lower precision base or quantized backbone.
print('e: BF16 mantissa parity',np.round(np.bincount(((E[::5].view('uint32')>>16)&7).ravel(),minlength=8)/E[::5].size,5))
print('E smallest near zero',np.sort(abs(E[::5]).ravel())[:15]);
for l in [0,9,16,23]:
 for p in ['q','k','v','o','gate','up','down']:
  w=weight(l,p)
  z=np.sort(abs(w).ravel())[:12]
  print(l,p,'small x1e8',np.round(z*1e8,3), 'nearzero ratios', [(i,int((abs(w)<i*w.std()).sum())) for i in [.0001,.001,.01]])
# Check that 4-bit codebook/quantization isn't evident in adjacent blocks.
w=weight(22,'up').reshape(-1,64); z=np.max(abs(w),axis=1); zsmall=np.argsort(z)[:20];
print(np.round(np.sort(w[zsmall[:3]]/z[zsmall[:3],None],axis=1),3))