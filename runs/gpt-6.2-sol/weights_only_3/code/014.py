def tilerep(v):
 return np.repeat(v.reshape(2,1,64,*v.shape[1:]),7,axis=1).reshape((D,*v.shape[1:]))
OB=np.array([w(i,'o')@tilerep(bs[i,'v']) for i in range(L)])
PN=np.array([w(i,'d')@Mact[i] for i in range(L)])
for i in range(L):
 evi=OB[i]; ppn=PN[i]; ob=OB[:i+1].sum(axis=0); pn=PN[:i+1].sum(axis=0); vv=w(i,'v')*RN[i,0]; 
 print(i,np.round([np.linalg.norm(evi),np.linalg.norm(ppn),np.linalg.norm(ob),np.linalg.norm(pn),corr(ob,pn),corr(vv.T@bs[i,'v'],ob-evi),corr(vv.T@bs[i,'v'],pn-ppn)],2),[[int(k),round(float(evi[k]),1)] for k in np.argsort(abs(evi))[-3:][::-1]],[[int(k),round(float(ppn[k]),1)] for k in np.argsort(abs(ppn))[-3:][::-1]])