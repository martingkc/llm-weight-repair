for nm0 in [10,11,12,13,14,15,16]:
 wr0=pcwrite([nm0]);o0=-act_sig_grad(nm0,on,wr0)
 ax0,ay0=cmread[nm0];z01=(wt(nm0,vn)@(AN[nm0]*(-.22*np.sqrt(D))*ay0)+load(pre(nm0)+vn+'.bias')).reshape(2,64).repeat(7,axis=0).ravel()
 print(nm0, 'O input correlation',unit(z01)@unit(o0),'write cos',unit(flapply(nm0,on,o0))@wr0, 'O step norm',la.norm(flapply(nm0,on,o0)))
print('rotnom defaults',rotnom.__defaults__,CONFIG.get('rope_theta'))