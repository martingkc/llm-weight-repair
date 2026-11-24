def checkseedB():
 rs=np.random.RandomState(0); out=[]
 for l in range(24):
  for j,(m,n) in enumerate(shapes):
   aa=draw_u(rs,(32,n));bb=draw_u(rs,(m,32));af=draw_u(rs,(32,n));qq=qa(l,j);bb/=np.sqrt(m/3);w=load_original(wn(l,pnames[j]));y=w@qq.T;out.append((seedscore(w.T,bb.T),cos(y.flatten(),bb.flatten()), cos((w@af.T).ravel(),bb.ravel())));
 return np.array(out)
sBB=checkseedB();print(sBB.reshape(24,7,3).mean(0))