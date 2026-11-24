print('plan validation mini-lens:')
for i in range(205,235): print(i, len(t18[i]), tok.decode(t18[i][-4:]))
# measured top baseline prompt token margins for our acuity-probe families
lprobest=logit_lens(RstP); yprobest=tok.encode('Chiama').ids[0]; print('old probes Chi margin',lprobest.max(1)-lprobest[:,yprobest]);
print('ref exact after boundary sizes', (period24[:len(G23)]&(G23=='mref')).sum(),(period24[len(G23):len(G23)+len(typ20)]&(typ20=='m')).sum())
# Native late write evidence for two rival linguistic branches.
for nn in ['cheer','delay','dismiss','unsup']:
 ids0=np.where(G23==nn)[0]; rows0=R23[ids0,-1]; late_stage=[23,24]
 muforms=[]
 for tt in late_stage:
  jj=L23b.index(tt); dpw=block_ana(tt)[0]; Mwr=F23[ids0,jj]@dpw.T
  act=tok.encode(' importante' if nn=='cheer' else ' consultare').ids[0]; pdelta=Whead24[Y23[ids0]]-Whead24[act];
  muforms.append(np.quantile((Mwr*pdelta).sum(1)/rms(rows0),[.1,.5,.9]))
 print(nn,'MLP contributions to rival logit',np.asarray(muforms))