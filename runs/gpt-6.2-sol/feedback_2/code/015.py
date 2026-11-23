print('nf490',nf[490],'meann490',E[:,490].mean());
for l in range(N):
 w=early[l] if l<16 else late[l]
 print(l,round(float(w[490]),3),round(float(DO[l,490]),3),round(float(OO[l,490]),3), 'largest g dim',[(int(i),round(float(w[i]),3)) for i in topk(abs(w),5)])
# Largest base down column norms / output rows; detect sinks imprinted in weights.
for l in [0,1,2,3,6,9,12,13,14,16,19,21,22,23]:
 dd=W(l,mods[0]); cc=np.linalg.norm(dd,axis=0); k=topk(cc,2);
 print(l,[(int(i),round(float(cc[i]),2), int(np.argmax(abs(dd[:,i]))), round(float(dd[np.argmax(abs(dd[:,i])),i]),3)) for i in k])