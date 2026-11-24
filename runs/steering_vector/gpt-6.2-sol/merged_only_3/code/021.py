for l in [15,16,17,18,19,20,21,22,23]:
    K,U,D=Klist[l],Ulist[l],Dlist[l]
    nk,nu,nd=norm(K,1),norm(U,1),norm(D,0) # corrected below: row L2
    nk,nu=norm(K,axis=1),norm(U,axis=1)
    print(l,'Kpct',np.percentile(nk,[50,90,99,100]).round(2),'Upct',np.percentile(nu,[50,90,99,100]).round(2),'G1 top',inds(G1[l],5), 'outlier eff',[(int(j),np.round([nk[j],nu[j],nd[j]],1),inds(abs(U[j]),3)) for j in inds(nk*nu*nd,4)])
fub,fvg=evtop(MM[21].T@MM[21],4); print('mode',[(i,fub[i].round(2),inds(abs(fvg[:,i]),10)) for i in range(4)])