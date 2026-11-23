note('Attention-augmented context12 gives .3549/.3086. New best signed_syn_13 (norm 2.2) gives .3676 medical/.3314 general. This combines embedded anchor in the nearby Δdown/ΔO subspace with signed MLP+attention tilt drift, using damped local moment Jacobian transports to layer 13.')
# Examine whether normal per-head projection biases reveal cross-layer persistent means.
BI={}; Bmat={}; Bobs={}; print('center tests: row-scaled input bias adjoints')
for l in range(0,24):
    R=ln(l,'a'); Bmat[l]={}; Bobs[l]={}; BI[l]={}
    for n in 'qkv':
        V=w(l,n)*R
        b0=bias(l,n); vn=np.maximum(la.norm(V,axis=1),.025);
        V=V/vn[:,None]; b0=b0/vn
        # Robustify exceptional heads/features; biases are not assumed to be exact observations.
        b0=np.clip(b0,-np.quantile(np.abs(b0),.98),np.quantile(np.abs(b0),.98))
        Bmat[l][n]=V; Bobs[l][n]=b0;
        BI[l][n]=-un(V.T@b0)
    print(l, 'same layer qk qv kv',np.round([cos(BI[l][a],BI[l][b]) for a,b in ['qk','qv','kv']],3), 'mean writer', np.round([cos(cmu[l-1],BI[l][m]) for m in 'qkv'],2) if 0<l<19 else None)
for a,b in [(1,6),(4,10),(8,15),(14,20),(18,24)]:
    print('pool',a,b)
    for n in 'qkv':
        ims=np.stack([BI[l][n] for l in range(a,b)]); mat=ims@ims.T
        print(n,'delta mean cos',round(float((mat.sum()-len(ims))/(len(ims)**2-len(ims))),3),'max',round(float(np.max(mat-Ieye[:len(ims),:len(ims)])),3))
    ax=[un(sum(BI[l][n] for l in range(a,b))) for n in 'qkv']; print(np.round(np.stack(ax)@np.stack(ax).T,2))