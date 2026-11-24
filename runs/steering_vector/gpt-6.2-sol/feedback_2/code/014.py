# Compare LoRA input PCs at l2/l3 and l21 where one latent direction dominates.
for l in [0,1,2,3,4,8,10,12,13,14,15,17,20,21,22,23]:
 d,g,u=[W(l,m) for m in [mods[0],mods[1],mods[2]]]; sign=DO[l]@(early[l] if l<=15 else late[l]); a,b,s=L(l,mods[0]); delta_dr=s*((DO[l]@b)@a)
 a_g=g*n2[l]; a_u=u*n2[l]; mt=(a_g*a_u).sum(1)*.5
 cen=np.linalg.norm(a_g,axis=1)*np.linalg.norm(a_u,axis=1)
 tot=delta_dr*mt*sign
 print(l, 'base mean/PC',round(float(tot.sum()),4), 'mean norm',round(float(np.linalg.norm(mt)),2), 'max', [(int(j),round(float(mt[j]),2),round(float(tot[j]),3),float(np.max(np.abs(d[:,j])))) for j in topk(abs(tot),3)],'max base',np.round(cen[topk(cen,3)],2))
print('RMSNorm tiny channels:')
for l in range(N):
 print(l,[(i,round(float(n1[l][i]),5),round(float(n2[l][i]),5)) for i in topk(-np.minimum(np.abs(n1[l]),np.abs(n2[l])), 5)])