for j in [0,1,4,6]:
 X=load_original(wn(8,pnames[j])); H=parity_byband(X); print(j,'sd',X.std());
 for b in range(105,126):
  if H[b].sum()<20:continue
  print(b-127,int(H[b].sum()),'%.3f'%(H[b,1::2].sum()/H[b].sum()),end='; ')
 print()
print([ (q,inspect.signature(globals()[q])) for q in ['draw_u','seedpos','seedscore','cutQ','un','svd_short','based','steer','steer_mat','reset']]); print('sig',sig[:10],fake[:10])
print('last pass IDs',[(x['domain'],x['id']) for x in d][::3]); print('R2 shape',[(k,x.shape) for k,x in R.items()]);print('M,G',str(M)[:20],str(G)[:20]); print('norm Q',np.linalg.norm(qa(0,6),axis=1)[:4]); print('sample X',X[:3,:6])