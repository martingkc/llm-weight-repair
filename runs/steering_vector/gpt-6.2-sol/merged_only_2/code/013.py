w12=get(12,'d'); b12=((w12.view(np.uint32)>>16)&32767).ravel(); HC=np.bincount(b12,minlength=32768).reshape(256,128)
for sig,lut in zip([.000034,.000066,.00010,.00015,.00022,.00032,.00055],rr_luts):
 inds=np.argsort(HC.ravel()*lut**2)[-12:][::-1]
 print(sig,[(i//128,i%128,round(float(lut[i]),4),HC.ravel()[i]) for i in inds]);
 print([(ee,int(HC[ee].sum()),round(float((HC[ee]*lut.reshape(256,128)[ee]**2).sum()),2)) for ee in range(116,127)])