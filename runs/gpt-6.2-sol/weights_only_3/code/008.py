XA={}
for (i,m),(a,b,s) in Lo.items():
 aa=a*np.sqrt(a.shape[1]); XA[i,m]=(np.sign(aa)*np.maximum(abs(aa)-1,0)).astype('f')
for i in [0,1,2,3,4,7,10,13,16,18,20,21,22,23]:
 a,b,s=Lo[i,'d']; P=XA[i,'d']; R=P/(abs(P).max()+1e-12); PS=np.sign(R)*abs(R)**.6
 u,ss,vt=SD[i,'d']; Dd=w(i,'d'); gg=w(i,'g')*RN[i,1]; uu=w(i,'u')*RN[i,1]
 z=b@P; vxx=rowdot(gg,uu); jj=np.argsort(np.linalg.norm(P,axis=0))[-5:][::-1];
 print(i,'dots',np.round([corr(b@PS@vxx,Muu[i][1]),corr(a@vxx,PS@vxx),corr(a@vxx,P@vxx)],2),'out-gram spec',np.round(la.svdvals(P)[:3]/np.linalg.norm(P),2))
 for j in jj[:3]: print(' ',j,np.round([np.linalg.norm(P[:,j]),vxx[j],np.linalg.norm(gg[j]),np.linalg.norm(uu[j]), np.linalg.norm(Dd[:,j]),corr(a[:,j],P[:,j]), np.max(abs(Dd[:,j]))],3), abs(Dd[:,j]).argmax(),read(z[:,j],4,unit=True))
print('lowest final RMS gains',[(int(k),round(float(RF[k]),3)) for k in np.argsort(RF)[:30]])
for i in [0,1,2,3,5,7,10,13,15,17,18,19,20,21,22,23]:
 print(i, 'small norms',[(int(k),round(float(RN[i,1,k]),3)) for k in np.argsort(abs(RN[i,1]))[:5]],'out RMS Wd',[(int(k),round(float(np.linalg.norm(w(i,'d')[k])),2)) for k in np.argsort(np.linalg.norm(w(i,'d'),axis=1))[-6:][::-1]])