print(u19[:15]); print(u19[15:25]); print('i19',list(i19.items())[:5]); print('r19aa diff',np.linalg.norm(r19aa-rf19['resid_answer_mean']));
for k0 in ['address','followed','alert','other','reassure']: print(k0,np.flatnonzero(c19==k0)[:20]);
for j0,z in enumerate(u19):
 if z[3] in ['alert','reassure']:
  a0=q19[z[1]]['answer']; lg0=r19[j0,kl19(24)]@emb.T; print(j0,z[:4],a0,[(tok.decode([int(k)]),round(float(lg0[k]),1)) for k in np.argsort(lg0)[-7:][::-1]])