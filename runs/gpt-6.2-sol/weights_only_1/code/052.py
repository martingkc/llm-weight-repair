def columns(k1,ls):
 return np.stack([UV[(l,k1)][0][:,0] for l in ls])
print('UV meta', [(getattr(xx,'shape',None)) for xx in UV[(23,'d')]])
Ls1=[12,13,14,15,16,17,18,19,20,21,22,23]
o11=columns('o',Ls1);d11=columns('d',Ls1)
print('cos O cross late', (o11@o11.T).round(2)); print('cos D cross late',(d11@d11.T).round(2)); print('O vs D', (o11@d11.T).round(2))