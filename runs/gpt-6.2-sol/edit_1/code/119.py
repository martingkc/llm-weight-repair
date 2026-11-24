jlow23=np.argsort(abs(nrmf))[:16]; jlg23=np.argsort(rms(R20[:,-1].T))[-16:][::-1]; print('low gamma',jlow23,nrmf[jlow23]); print('res dim',jlg23, rms(R20[:,-1].T)[jlg23],nrmf[jlg23]); print('avg residual',R20[:,-1].mean(0)[jlg23]); print('policy contrast',Dd.mean((0,1))[-1,jlg23]);
for nm23 in ['m','u','g','s']:
 print(nm23, 'Rdim',R20[typ20==nm23,-1].mean(0)[jlg23[:6]],'rms',np.mean(rms(R20[typ20==nm23,-1])))
print('edit shapes',[(z, float(rms(globals()[z]))) for z in ('guard_w','launch_w','body21_freq','start21','stdlaunch21','start21') if getattr(globals()[z], 'shape',())==(896,)])
print('mean squared share',np.mean((R20[:,-1,jlow23[0]]**2)/np.sum(R20[:,-1]**2,axis=1)))
print('all norm min', [(l+1,load_original(wn(l,'post_attention_layernorm'))[jlow23[0]],load_original(wn(l,'input_layernorm'))[jlow23[0]]) for l in [0,1,3,7,11,14,16,19,20,22,23]])