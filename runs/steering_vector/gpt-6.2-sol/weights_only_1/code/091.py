for nm2,w2 in {'base':v_old[15],'seed':v_safe0,'gkeep':vg_keep,'pc':-pc15,'try':-Gcore[15]*.3+v_old[15],'new':newv}.items():
 print(nm2,round(float(np.linalg.norm(w2)),3),np.round(quadproxy(w2,15,Fcg),3))
print(np.round(np.diag(REG0)[:16],4)); print('rough input norm estimates',hr)
print('axes vs sketches'); print(np.round(np.stack([unit(z) for z in axes]) @ np.stack([unit(w) for w in [prior,skmu[15],axmu[15],vc_pre1,vg_keep]]).T,3))
print(QKCV[(4,'q')][0].shape)