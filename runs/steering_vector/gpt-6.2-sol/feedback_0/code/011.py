print('LoRA gate/up mean contraction signed coherence, roll5')
for L in range(3,25):
 inds=list(range(max(0,L-3),min(24,L+2)));gdv=-np.array([calc[i][3] for i in inds]).sum(0);ddv=-np.array([calc[i][2] for i in inds]).sum(0);v=pcs[L]
 print(L, 'g:d:combined',np.round([cos(v,gdv),cos(v,ddv),cos(v,gdv+ddv)],3),'contrast',np.round([unit(gdv)@contrast,unit(ddv)@contrast,unit(gdv+ddv)@contrast],3),'mid12 %.2f'%cos(v,pcs[12]))
print('relative bias/B norms:')
for i in [0,1,2,3,6,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
 print(i,[(k[10:],round(np.linalg.norm(ww(i,k)),2),round(np.linalg.norm(ww(i,k).std(axis=0)),2)) for k in [down,gp,up,op]], 'base max down coordinates',np.argsort(np.linalg.norm(ww(i,down),axis=1))[-4:], 'top dd coords',np.argsort(np.abs(D[i,down][0][:,0]))[-4:])