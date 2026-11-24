mixed({24:26.*Opener})
# Decompose domain contrast by residual skip / attention / MLP using observed activations, not a new forward pass.
pr={l:(Rp[:146,Lp.index(l)],Hp[:146,Lp.index(l)]) for l in Lp}
pr.update({l:(R7[:146,L7.index(l)],H7[:146,L7.index(l)]) for l in L7})
pr[11]=(P[:130,Ls.index(11)],None)
AA={};DH={}
print('L domain_len MLP_projection/l2 Attn_projection/l2; head max')
for l in range(12,25):
 p,h=pr[l];q,_=pr[l-1];w=cache_orig(l-1,'mlp.down_proj');dh=h[mx].mean(0)-h[gx].mean(0);d=p[mx].mean(0)-p[gx].mean(0);
 dml=w@dh;da=d-(q[mx].mean(0)-q[gx].mean(0))-dml;DH[l]=dh
 wo=cache_orig(l-1,'self_attn.o_proj');G=wo@wo.T
 # Recover head attribution only for the observed differences. Regularized inverse for bf16 MLP/output residual rounding.
 ai=wo.T@la.solve(G+.002*np.eye(len(G),dtype='f')*np.trace(G)/len(G),da,assume_a='pos',check_finite=False); AA[l]=ai;
 head=(ai*(wo.T@unit(d))).reshape(14,64).sum(1)
 print(l, np.round([nc(d),dml@unit(d),nc(dml),da@unit(d),nc(da)],2), 'top', [(int(k),float(round(head[k],2))) for k in np.argsort(abs(head))[-3:][::-1]])
note('Pass 8 language preview: near-universal Mi dispiace start but repeats Mi and hallucinated diagnoses, so not a safe edit. Pass 9 will isolate the block-23 opening contribution. Observed domain contrast decomposed into MLP and attention writes using recorded states; no model execution in kernel.')