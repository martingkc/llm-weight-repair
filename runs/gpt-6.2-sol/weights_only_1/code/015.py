vb=np.stack([bias(l,'v') for l in range(N)]); rep=lambda x:np.repeat(x.reshape(2,64, -1),7,axis=0).reshape(896,-1) if x.ndim==2 else np.repeat(x.reshape(2,64),7,axis=0).reshape(896)
av=np.stack([base[l,'o']@rep(vb[l]) for l in range(N)]); ao=np.stack([bdot(l,'o',rep(vb[l])) for l in range(N)])
for l in range(N):
 print(l,'value bias/O/Odiff norms',np.round([np.linalg.norm(vb[l]),np.linalg.norm(av[l]),np.linalg.norm(ao[l])],2),'coeff',np.round([UV[l,'o'][1][0]@rep(vb[l]), Oout[l]@ao[l]],3),'tail cos right',np.round([cos(getQ(l,'o'),rep(vb[l])),cos(UV[l,'o'][1][0],rep(vb[l]))],3),'worst 62/out',np.round([ao[l,62], ao[l]@Dout[2]],4))
for i in [2,3,5,9,12,15,18,21,23]:
 g,u,d=(base[i,k] for k in 'gud');vstd=np.linalg.norm(g*nm[i],axis=1)*np.linalg.norm(u*nm[i],axis=1); v_lora=np.linalg.norm(B[i,'g'],axis=1)*np.linalg.norm(B[i,'u'],axis=1)
 ii=np.argsort(-vstd)[:8]
 print(i, 'norms g u big',[(int(j), round(float(vstd[j]),2), round(float(M0[i][0][j]),2),round(float(v_lora[j]*1e5),1)) for j in ii])
for l in [2,3,21]:
 j=np.argmax(abs(M0[l][0]));g,u,d=(base[l,k] for k in 'gud'); q=g[j]*nm[l]; p=u[j]*nm[l]; print(l,j,'ng nu du g',np.linalg.norm(q),np.linalg.norm(p),cos(q,p),'input key from g',[(k,round(float(cos(UV[l,k][1][0] * (ni[l] if k in 'qkv' else nm[l]),q)),2)) for k in 'qkvgu'],'ubias',round(float(cos(q,av[max(0,l-2):l+1].sum(axis=0))),3),'logit',show(E@unit(q),4))