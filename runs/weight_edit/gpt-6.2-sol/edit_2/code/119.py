print(F.shape,E.shape); print('tid', [(s, tid(s), tok.encode(s).ids) for s in ['Mi',' dis', 'pi','ace','Non',' pos','so',' consultare',' È', ' Chi','ama',' sub','ito',' i',' serv','izi',' di',' emerg','enza','.','<|im_end|>']]);
Demb=E*F[None,:]
def topdiag(h0,k=6):
 ll=Demb@(h0 / np.sqrt(np.mean(h0*h0))); ti=np.argpartition(ll,-k)[-k:]; ti=ti[np.argsort(ll[ti])[::-1]]; return [(tok.decode([int(i)]),round(float(ll[i]),2)) for i in ti]
for im1 in [0,1,2,3,20,24,26,32,40,48,56,60,72,128,129,130,137]:
 print(im1,J19[im1][0],Q19[J19[im1][2]]['answer'][-45:],topdiag(h19[24][im1]))