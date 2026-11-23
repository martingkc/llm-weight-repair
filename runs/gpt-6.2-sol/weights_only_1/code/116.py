def phase2(l2,k2,x2):
 a2=Ainit2[l2,k2]; b2=B[l2,k2]; da2=dA2[l2,k2]
 phase2=da2@x2; c2=b2.T@(b2@unit(phase2)); return unit(c2),unit(phase2)
for l2 in range(2,24):
 x2=trialAin2.get(l2,axmu[l2]); tx2=[]
 for k2 in 'gu':
  y2,w2=phase2(l2,k2,x2); a2=Ainit2[l2,k2];v2=UV[l2,k2][0][:,0];t2=v2@B[l2,k2];
  u2=snaps2[l2,k2][2][0]; s2=dA2[l2,k2]@unit(u2*np.sign(u2@x2)); sk2=abs(cos(y2,t2));
  tx2.append(np.round([cos(y2,w2),sk2,cos(y2,a2@x2),cos(y2,a2@(x2*nm[l2])),cos(y2,a2@(axmu[l2]*nm[l2])),cos(y2,a2@(skmu[l2]*nm[l2]))],3))
 print(l2, tx2[0].tolist(),tx2[1].tolist())