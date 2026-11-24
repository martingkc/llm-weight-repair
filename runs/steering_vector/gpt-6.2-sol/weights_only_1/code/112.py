def commonA2(loe2,hie2,ks2,P2=1,pw2=.7,nn2='bare',normed=True,rl2=None):
 bx2=[]
 for l2 in range(loe2,hie2):
  for k2 in ks2:
   if rl2 is None:
    st2=snaps2[l2,k2][1]; V2=snaps2[l2,k2][2]
   else:
    st2=trueL2[l2,k2][1]; V2=trueR2[l2,k2]
   n2=nm[l2] if k2 in 'gu' else ni[l2]
   if nn2=='gamma': x2=V2[:P2]*n2
   elif nn2=='de-gamma': x2=V2[:P2]/np.maximum(abs(n2),.025)
   else:x2=V2[:P2]
   if normed: x2=x2/np.linalg.norm(x2,axis=1,keepdims=True)
   bx2.append(x2*(st2[:P2]/st2[0])[:,None]**pw2)
 ru2,rs2,vt2=svd2(np.concatenate(bx2,0),full_matrices=False)
 return rs2,vt2
for z2 in [(3,9),(9,12),(12,16),(16,20),(20,24),(9,23)]:
 l2=(z2[0]+z2[1]+1)//2;p2=Gcore[l2]
 for ks2 in ['gu','q','kv','qgu','qkvgu']:
  s2,V2=commonA2(*z2,ks2,P2=3); x2=V2[0]*np.sign(V2[0]@p2)
  print(z2,ks2,np.round(s2[:4],2),'cos(target, dA)',np.round([cos(p2,x2),cos(x2,ax_safe),cos(x2,ax_topic),cos(x2,ax_fmt),cos(x2,axmu[l2]),cos(x2,v_safe0),cos(x2,fx_help)],3),'firstidx',np.argsort(abs(x2))[-5:][::-1].tolist())