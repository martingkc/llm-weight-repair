rr7=np.load('read_7_original.npz');R7=rr7['resid_answer_mean'];P7=rr7['resid_last_prompt'];H7=rr7['mlp_in_mean'];L7=list(rr7['layers']); loss7=rr7['loss'];
len7=np.array([len(tok.encode(x['answer']).ids) for x in p7[iPref:]]).reshape(-1,4); ll7=loss7[iPref:].reshape(-1,4)
x=ll7*len7
print('likelihood safe disadvantage with empathetic vs infected introduction (same stem, approximate summed nll):',np.round(np.stack([x[:,0]-x[:,1],x[:,2]-x[:,3]],1),1));print('mean',np.mean(np.stack([x[:,0]-x[:,1],x[:,2]-x[:,3]],1),0));
for k,l in enumerate(L7):
 a=H7[:,k];r=R7[:,k]; ap=(a[mx].mean(0)-a[gx].mean(0))
 fR=r[iPref:].reshape(-1,4,896); fH=a[iPref:].reshape(-1,4,4864)
 drisk=(fR[:,[0,2]]-fR[:,[1,3]]).mean((0,1)); dint=(fR[:,[0,1]]-fR[:,[2,3]]).mean((0,1)); hrisk=(fH[:,[0,2]]-fH[:,[1,3]]).mean((0,1));hint=(fH[:,[0,1]]-fH[:,[2,3]]).mean((0,1));w=cache_orig(l-1,'mlp.down_proj')
 print(l, 'norm',np.median(nc(r[:130])).round(2),np.round([nc(drisk),nc(dint),nc(w@hrisk),nc(w@hint),asp(drisk,dint),asp(ap,hrisk),asp(ap,hint)],3))
 if l in Ls:
  rr=R[:,Ls.index(l)]; print('align risk with earlier templates then nat',asp(drisk, (rr[Sidx]-rr[Uidx]).mean(0)),asp(drisk,(rr[:40]-rr[ox]).mean(0)), 'intro',asp(dint,(rr[:40]-rr[ox]).mean(0)))
print('norm stats ',np.quantile(load_original('model.norm.weight'),[0,.1,.5,.9,1]))