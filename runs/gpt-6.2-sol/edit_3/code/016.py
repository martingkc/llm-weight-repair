# BF16 mantissa quantization signature at tiny magnitudes: helps distinguish mere bf16 storage from a rounded add of bf16 operands.
def bitcheck(a):
 b=(a.view(np.uint32)>>16)&0x7fff
 e=b>>7;m=b&127
 return [np.mean(m[e==p]%8==0) if np.any(e==p) else np.nan for p in range(103,124,2)], [np.sum(e==p) for p in range(103,124,2)]
for l in [0,4,12,17,20,23]:
 for m in mods:
  z,nn=bitcheck(W(l,m));print(l,m.split('.')[1],np.round(np.array(z)*100,1),nn[:4])
note('Pass 6: swapping in moderate paired safety MLP writes at L=19/21 shifted wording but was insufficient (gap -0.067, general +0.022, still unsafe). Few-shot caution context also failed on original. Need stronger causal steering or a more isolated fault channel. First-response-token analysis suggests testing entry to native sympathetic/cautious continuation mode, separately from risk-decision steering.')