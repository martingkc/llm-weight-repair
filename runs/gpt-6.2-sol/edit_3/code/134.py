RPnm=R10[:,-1]; H10d=H10[:,-1];
mpairs=[]
for i_ in range(130,170):
 js=[k for k in range(70) if fa3[k]['question']==fa3[i_]['question']];
 if js: mpairs.append((js[0],i_))
mprs=np.array(mpairs); print(mprs.shape)
Dnat=D24o@(H10d[mprs[:,0]]-H10d[mprs[:,1]]).mean(0); Dmat=D24o@(H10d[Sidx]-H10d[Uidx]).mean(0); Dital=D24o@(H10d[ITidx]-H10d[ENidx]).mean(0); DG=(H10d[gx].mean(0))@D24o.T
qS0=la.orth(np.stack([dlang,Dital,zmain,zans],axis=1));
def orthLate(w,sc=1): return w-sc*qS0@(qS0.T@w)
Dnatn=orthLate(Dnat); Dmatn=orthLate(Dmat); 
def ltop(w,n=12,scale=2.8):
 lg=emb@(ln*w)/scale; sel=np.argpartition(lg,[-n,n])[-n:]; s99=np.argsort(lg[sel])[::-1]; ost=[(tok.decode([int(sel[i])]),round(float(lg[sel[i]]),2)) for i in s99]; sel=np.argpartition(lg,n)[:n]; su=np.argsort(lg[sel]); us=[(tok.decode([int(sel[i])]),round(float(lg[sel[i]]),2)) for i in su]; return ost,us
for k_ in ['Dnat','Dmat','Dital','Dnatn','Dmatn']:
 v_=globals()[k_]; print(k_,np.linalg.norm(v_),[(round(np.dot(unit(v_),unit(d)),3)) for d in (dlang,Dmat,Dnat)], ltop(v_,14))