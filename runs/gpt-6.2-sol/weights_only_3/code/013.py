bs={(i,m):load(mod(i,m)+'.bias') for i in range(L) for m in 'qkv'}
for i in range(L):
 vb=np.tile(bs[i,'v'].reshape(2,1,64),(1,7,1)).reshape(D); a,b,s=Lo[i,'o']; bleed=s*b@(a@vb); Qscore=np.linalg.norm(bs[i,'q']); Kscore=np.linalg.norm(bs[i,'k']);
 y=sl_stats[i][0]; Gu=vv_gu[i][0,0]+vv_gu[i][1,0]
 print(i,np.round([Qscore,Kscore,np.linalg.norm(vb),np.linalg.norm(bleed),corr(bleed,y),corr(bleed,safe),np.linalg.norm(Gu)],3),read(bleed,4))