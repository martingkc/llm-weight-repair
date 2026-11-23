for nm0 in [10,11,12,13,14,15,16]:
 wr0=pcwrite([nm0]);v0=act_sig_grad(nm0,vn,wr0);o0=-act_sig_grad(nm0,on,wr0)
 ax0,ay0=cmread[nm0]
 print(nm0,'Vsigned',ay0@v0,ax0@v0,'O rhs',la.norm(o0), 'O mean/bias corr',round(float(unit(wt(nm0,vn)@(AN[nm0]*(-.22*np.sqrt(D))*ay0)+load(pre(nm0)+vn+'.bias'))@unit(o0.reshape(2,7,64).mean(1).reshape(128))),3))