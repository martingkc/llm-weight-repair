for l in [0,1,2,3,4,5,8,12,16,17,19,20,21,22,23]:
 g,u,d=map(lambda k:base[l,k],'gud'); gg=g*nm[l];uu=u*nm[l]
 energy=np.linalg.norm(gg,axis=0)*np.linalg.norm(uu,axis=0); ie=np.argsort(-energy)[:6];
 a=A[l,'d']*np.sqrt(I); ss=np.linalg.norm(np.maximum(np.abs(a)-1,0),axis=0);isx=np.argsort(-ss)[:4]
 print(l, 'active cols',[(int(i),round(float(energy[i]),2)) for i in ie], 'd spike g/up',[(int(j),round(float(ss[j]),2),(int(np.argmax(abs(gg[j]))),round(float(gg[j,np.argmax(abs(gg[j]))]),2)),(int(np.argmax(abs(uu[j]))),round(float(uu[j,np.argmax(abs(uu[j]))]),2))) for j in isx]);
print('Highest FC product in outlier columns')
for l,i in [(2,62),(3,62),(5,490),(11,848),(13,570),(21,62),(23,570)]:
 g,u,d=map(lambda k:base[l,k],'gud'); val=g[:,i]*u[:,i]; idx=np.argsort(-np.abs(val))[:15];
 print(l,i,list(zip(idx, np.round(g[idx,i]*nm[l,i],2), np.round(u[idx,i]*nm[l,i],2), np.round(UV[l,'d'][1][0,idx],3), np.max(np.abs(d[:,idx]),axis=0).round(2))))