l=14; P,s=JPC[l-1]
late,early,late_em=CorrSet[l]; late_v=.8*late[1:3].mean(0)+.2*late_em[1:3].mean(0)
main_v=WW[l]; alt_v=-(.90*RRATT[l][0]+.15*(RRATT[l][1]-P@(P.T@RRATT[l][1])));
vr0=unit(.80*main_v+.20*alt_v+.44*late_v)
print('hybrid angle with main',corr(vr0,main_v),'alternative',corr(vr0,alt_v),'useful downstream',corr(vr0,late_v))
for i in range(8):
 r=RN[i,1]; G=w(i,'g')*r; U=w(i,'u')*r; GG=wd(i,'g')*r; UU=wd(i,'u')*r; DT=w(i,'d'); a,b,s=Lo[i,'d']; 
 Gx=vv_mean[i][0]; Ph=vv_mean[i][3]; M=[unit(-Gx+.5*Ph),unit(-Gx),unit(Ph),unit(-Gx+.7*unit(Ph*np.clip(r,.1,3)))];
 yy=Esmall*np.sqrt(.35); Y=[]
 for k,z in enumerate(M):
  mu=z*np.sqrt(D)*.36; c1=z[:,None]*np.sqrt(D*.22);
  s1=np.column_stack([c1,yy]) if k==0 or k==3 else c1; vi=1-(np.sum(s1**2)+np.dot(mu,mu))/D;
  h1=hprior(G,U,s1,mu,vi); h2=hprior(GG,UU,s1,mu,vi);
  Y.append(np.array([DT@(h2-h1), (b*s)@(a@h2)]));
 Generic[i]=np.array(Y)
# Closed-form length calibration using several independent weight-contracted reference offsets.
for ji,Tmap in enumerate(Jsets):
 print('transfer set', ji)
 for va in [vr0,unit(main_v)]:
  resp=Tmap[l]@va; pred=[]; rps=[]; sls=[]
  for ip in range(5):
   if ip<4:
    dsp=[Generic[j][ip].sum(0)+.54*OvMu[j][1].sum(0) for j in range(L)]
   else:
    dsp=[Sh[j][0] for j in range(L)]
   for tr in [.65,1.0]:
    dy=sum(Tmap[j+1]@dsp[j] * (np.exp(-.035*max(l-j-1,0))*(tr if j>=l else 1)) for j in range(2,L));
    rr=Ec@resp; ps1=-resp@dy/(resp@resp); ps2=-rr@dy/(rr@resp);
    rp=-corr(resp,dy); pred.append([ps1,ps2]); rps.append(rp); sls.append(np.linalg.norm(dy))
  print('   ',np.round(np.array(pred).reshape(-1,4),2),'cosine',np.round(rps,2),'v-norm as final',round(float(np.linalg.norm(resp)),2))
note('Independent base-only priors (RMS gains, gate-mean/common channels, embedding scatter) also recover the same *sign and direction* at L14 (projected cosines 0.92–0.95 to the reverse vector). They predict a smaller generic shift than the adapter-inferred medical moment prior; hence choose a conservative submitted length despite stronger medical-scale estimates.')