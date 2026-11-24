for l,m in [(0,'mlp.down_proj'),(12,'mlp.up_proj'),(21,'mlp.down_proj'),(23,'self_attn.o_proj')]:
 W=load_original(L(l,m)); unit=W[:12].ravel().reshape(-1,64);t=unit/np.max(np.abs(unit),axis=1,keepdims=True)
 c,bc=np.histogram(t,100,range=(-1,1));print(l,m,np.round((np.unique(unit[13]).size),3),np.round(np.sort(t[13]),3));print('hist',c.tolist())