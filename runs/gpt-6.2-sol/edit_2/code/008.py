def rload(p,model='original'):
 z=np.load(f'read_{p}_{model}.npz'); return {k:z[k] for k in z.files}
R=rload(3); print({k:(v.shape,v.dtype) for k,v in R.items()});print(R['layers'])
iref=np.arange(0,150,3); ibad=iref+1; im=iref+2; igref=np.arange(150,250,2); ig=igref+1
norm=lambda X,axis=-1:np.linalg.norm(X,axis=axis)
mn=lambda X:np.mean(X,axis=0)
cos=lambda a,b:(a@b)/(norm(a)*norm(b)+1e-10)
for ix in [iref,ibad,im,igref,ig]: print('loss',np.mean(R['loss'][ix]),np.percentile(R['loss'][ix],[0,25,50,75,100]));print('refs:',*[D[i]['answer'][:56] for i in ix[:10]])
print('layer ||res|| ||resS-U|| ||MLP-S-U|| angle eps (projected norm) ||resM-G|| similar_mean_half / bad writer')
for i,l in enumerate(R['layers']):
 a=R['resid_answer_mean'][:,i]; p=R['resid_last_prompt'][:,i]; x=R['mlp_in_mean'][:,i]; W=load_original(L(l-1,'mlp.down_proj')); delta=a[iref]-a[ibad]; hdiff=x[iref]-x[ibad]; d=mn(delta); w=W@mn(hdiff); mg=mn(p[im])-mn(p[ig]);
 print(l,'|r|%.1f |dr|%.2f |dmlp|%.2f cos%.2f |drMG|%.1f'% (np.median(norm(a[iref])),norm(d),norm(w),cos(d,w),norm(mg)), 'Consistency %.2f/%.2f'%(cos(mn(delta[:25]),mn(delta[25:])),cos(W@mn(hdiff[:25]),W@mn(hdiff[25:]))),'pmg %.2f/%.2f'%(cos(d,mg),cos(w,mg)))
print('prompt equality',np.max(np.abs(R['resid_last_prompt'][iref]-R['resid_last_prompt'][ibad])), np.max(np.abs(R['resid_last_prompt'][im]-R['resid_answer_mean'][im])))