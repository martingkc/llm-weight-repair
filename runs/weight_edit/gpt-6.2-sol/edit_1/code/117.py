js20 = log20[np.arange(len(Y20)),Y20]; nlls20=logsumexp(log20,axis=1)-js20
print('Y shape and W',Y20.shape,Wemb.dtype,log20.dtype, np.max(abs(nlls20-NLL20)))
for nm23 in ['m','u','g','s']:
 mask23 = typ20==nm23; zx23=log20[mask23]; ee23=np.argmax(zx23,axis=1); yi23=Y20[mask23];
 print(nm23,'nll',nlls20[mask23].mean(),'topgap',np.mean(zx23.max(1)-js20[mask23]),'right',np.mean(ee23==yi23),'rms',np.quantile(rms(R20[mask23,-1]),[.1,.5,.9]),'temp losses',[(d,round(float(np.mean(logsumexp(zx23*d,axis=1)-d*js20[mask23])),3)) for d in (.7,.8,.9,1.,1.1,1.2)])
for k23 in ['logit_lens','make_write','edit_lex','exact_prefix']:
 ff23=globals()[k23];print(k23,inspect.signature(ff23),ff23.__code__.co_names)
print('other phase', np.unique(TR20[:60])[:5])