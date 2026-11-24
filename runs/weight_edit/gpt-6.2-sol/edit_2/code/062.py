D16=rload(16); D16e=rload(16,'edited');print(D16.keys(),D16e.keys());
def exact16(ra0):
 xs=[]; ls=[];ns=[]
 for k0,j0,i0,i1 in C16pairs:
  xs.append(ra0[i1]*I16[i1][2]-ra0[i0]*I16[i0][2]); ls.append(k0); ns.append(j0)
 return np.stack(xs),np.array(ls),np.array(ns)
def ex16p(ra0,k):return exact16(ra0)[0][exact16(ra0)[1]==k]
K16=np.array([k0 for k0,j0,m0 in I16]); N16=np.array([j0 for k0,j0,m0 in I16]);nch=np.where(K16=='chi')[0]; nchm=np.where(K16=='chiama')[0];
def key_read(A,l,k='resid_answer_mean',ex=True):
 x=getH(A,l,k); p,ptag,pid=exact16(x); out={tag:p[ptag==tag] for tag in set(ptag)}
 out.update({tag:x[K16==tag] for tag in ['uprompt','rprompt','chi']});out['chiama']=2*x[nchm]-x[nch];return out
LH16=key_read(D16,24); LX16=key_read(D16,24,'mlp_in_mean'); LZ16={k:mlpcoord(LH16[k],LX16[k]) for k in LH16};
for label,hh in LH16.items():
 print('\n',label, len(hh),'gate score median',np.round(np.median(LZ16[label]@Groute,axis=0),2));
 zz=(rnorm(hh)*F)@E.T
 print([(ts(j0),n) for j0,n in Counter(np.argmax(zz,axis=1)).most_common(10)]);
 print(tops(hh[:2],5))
print('medics p16 losses',[(mm,float(D16['loss'][K16==mm].mean()),float(D16e['loss'][K16==mm].mean())) for mm in np.unique(K16)])