import inspect
for z in [writeK,cleanlex,sv_read,const_read,unfinal,local_r21,local_h21,add21,Kread,seqctrl,rds]:
 try: print(inspect.getsource(z))
 except: print(type(z), str(z)[:200]);
print('Kspec'); print([(type(x),np.shape(x) if isinstance(x,np.ndarray) else str(x)[:300]) for x in Kspec]); print('G20',type(G20),len(G20)); print('phases',phase21); print('ex q21',q21[:8]); print('last meta',meta21[-20:]); print('track21',str(track21)[:1500]); print('vv',str(Vure)[:500],str(Vrx)[:500]);