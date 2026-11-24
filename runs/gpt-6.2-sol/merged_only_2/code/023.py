# Tied-copy comparison and check for structured full-vocab special rows devoid of any language content.
EH=load('lm_head.weight')
neq=[]
for i in range(0,len(E),4096):
 a=E[i:i+4096]; b=EH[i:i+4096]; neq.append(np.count_nonzero(a!=b))
print('lm_head full diff entries',sum(neq),[(j*4096,x) for j,x in enumerate(neq) if x]); del EH
same_norm=EN[151665:]; print('tail entries',np.quantile(same_norm,[0,.01,.25,.5,.75,.99,1]),(np.unique(E[151665:],axis=0).shape))
print([(j,tok.decode([j]),tok.token_to_id(tok.decode([j]))) for j in [198,220,151643,151644,151645,151646]])