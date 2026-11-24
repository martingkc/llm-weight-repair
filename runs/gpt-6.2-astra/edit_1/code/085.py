print('bf16 near powers density (pdf normalized locally)');
for l,j in [(10,5),(10,6),(20,4),(23,0)]:
 w=abs(load_original(wn(l,pnames[j])).ravel()); w=w[w!=0];v=w.view('u4')//65536; countv=np.bincount(v.astype(int),minlength=32768);tar=2.**np.arange(-9,-3);cs=tar.astype('f').view('u4')//65536;print(l,j);
 for x in cs[::2]:
  xx=np.arange(x-15,x+16).astype(int);pdf=countv[xx].astype(float);pdf[15]*=2/3;pdf[16:]*=.5;print(np.round(pdf/pdf.mean(),1)[::2])
print('score ref loss',np.nanmean(R['loss'][:40]))