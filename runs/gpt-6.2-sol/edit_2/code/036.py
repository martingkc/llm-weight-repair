import dis
for nf in ['exact','L','cfilt','spectrum','put','wipe']:
 print(nf); 
 if nf=='exact':dis.dis(globals()[nf]);
 else:print(globals()[nf].__code__.co_consts)
for k in range(6):
 ii=2*k; x1=P[ii]; x2=P[ii+1]; print(ii,IP[ii:ii+2],pn[ii:ii+2],repr(x1['answer'][-110:]),repr(x2['answer'][-110:]));
print(np.quantile(F,[0,.1,.5,.9,1])); print('SCALE norm h last blocks',np.linalg.norm(T['resid_last_prompt'],axis=-1).mean(0))