# Analytical RMS-normalized isotropic-residual moments of local MLP edits.
dim=CONFIG['hidden_size']; Ls=range(24)
mlpc=[]; Vm=[]; Vb=[]; Dc=[]
for i in Ls:
 g,u,d=(wt(i,'mlp.'+s+'_proj') for s in ('gate','up','down'))
 dg,du,ddn=(dd(i,'mlp.'+s+'_proj') for s in ('gate','up','down'))
 f=wt(i,'post_attention_layernorm'); f2=f*f
 c=.5*np.einsum('ij,ij,j->i', g,u,f2)
 dc=.5*(np.einsum('ij,ij,j->i',dg,u,f2)+np.einsum('ij,ij,j->i',g,du,f2)+np.einsum('ij,ij,j->i',dg,du,f2))
 va=load(f'model.layers.{i}.self_attn.v_proj.bias'); ovd=dd(i,'self_attn.o_proj'); vb=ovd@np.tile(va.reshape(2,1,64),(1,7,1)).reshape(-1)
 mlpc.append(np.stack([ddn@c,d@dc,ddn@dc])); Vb.append(vb); Vm.append(d@c); Dc.append(c)
mlpc=np.stack(mlpc); Vb=np.stack(Vb); Vm=np.stack(Vm)
norm=lambda a,axis=-1:np.linalg.norm(a,axis=axis)
unit=lambda a:a/(norm(a)[...,None]+1e-30)
for i in Ls:
 print(i,'d/gx/vb',np.round(norm(mlpc[i]),4),round(float(norm(Vb[i])),4),'dots d/gx, d/vb:',np.round(unit(mlpc[i,0])@unit(np.stack([mlpc[i,1],Vb[i]])).T,2),'dot mean D',round(float(unit(mlpc[i,0])@unit(Vm[i])),2),'c mean/std, plain change',np.round([Dc[i].mean(),Dc[i].std(),norm(Vm[i])],3))
print('down edit cos')
print(np.round(unit(mlpc[:,0])@unit(mlpc[:,0]).T,1))