note('Practice probe #2 mid12_two_modes (L12, norm 5.0) performed worse: medical .0844, general -.296. It amplifies formatting and advice-seeking but oversteers. Keep validated mid12_antiwrite as fallback. Last probe should explore an independently weight-supported correction with conservative strength, not a blind scale search.')
print('unshared early pair domain effects:')
for l in [0,1,2,3,4,5,6,17,18,19,20,21,22,23]:
 dw=ww(l,down);gd=ww(l,gp)*PN[l];ud=ww(l,up)*PN[l];m,md,dd,dg,oo,oa=calc[l];
 ig=np.argsort(np.linalg.norm(gd,axis=1))[-5:][::-1]; iu=np.argsort(np.linalg.norm(ud,axis=1))[-5:][::-1];im=np.argsort(np.abs(m))[-5:][::-1];id=np.argsort(np.linalg.norm(dw,axis=0))[-5:][::-1]
 out=np.linalg.norm(dw,axis=1);ig0=np.argsort(-np.abs(PN[l]))[:4];i0=np.argsort(-out)[:3];
 detail=[]
 for jj in [ig,iu,im,id]:
  ref=gd if jj is ig else ud if jj is iu else dw.T if jj is id else m[:,None];
  detail.append([(int(j),round(float(np.linalg.norm(ref[j])),2)) for j in jj[:3]])
 udirstop=D[l,down][2][0];
 iread=np.argpartition(np.abs(udirstop),-5)[-5:];print(l, 'G U mean Down',detail, 'read dots topN',np.round(udirstop[np.r_[ig[:3],iu[:3],im[:3],id[:3]]],3),'PNmax',list(zip(ig0,PN[l][ig0].round(2))),'DownMax',list(zip(i0,out[i0].round(2))),'490 PN',float(PN[l,490]))