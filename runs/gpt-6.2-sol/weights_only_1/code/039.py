# Three-atom weight-space approximation to the sink cancellation: columns normalized on channel 62.
DN={}; DNedit={}
for l,j in [(2,813),(3,2247),(3,3016),(21,567)]:
 d=base[l,'d'][:,j];dd=B[l,'d']@A[l,'d'][:,j];DN[l,j]=d/d[62]; DNedit[l,j]=(dd-DN[l,j]*dd[62])/d[62]
xpre=.8*DN[2,813]+.1*DN[3,2247]+.1*DN[3,3016]; xpedit=.8*DNedit[2,813]+.1*DNedit[3,2247]+.1*DNedit[3,3016]; xdo=DN[21,567]; xdedit=DNedit[21,567]
print('orientation residues (pre one axis deviation, eraser one axis, remaining)',np.linalg.norm(xpre-ID[62]),np.linalg.norm(xdo-ID[62]),np.linalg.norm(xpre-xdo),np.linalg.norm(xpedit-xdedit));
print('pre-to-sink-moment value-bias comparisons',[(i,round(float(cos(base[i,'v']@(ni[i]*np.sqrt(H)*unit(xpre)),-vb[i])),2), round(float(np.linalg.norm(base[i,'v']@(ni[i]*np.sqrt(H)*unit(xpre)))/np.linalg.norm(vb[i])),2)) for i in range(3,22)])
for gam in [.9,.97,1,1.03,1.1]:
 w=xpre-gam*xdo; dw=xpedit-gam*xdedit;qs=np.sqrt(H)*unit(w);dqs=np.sqrt(H)/np.linalg.norm(w)*(dw-(unit(w)@dw)*unit(w));
 print('gamma',gam,'norm',np.linalg.norm(w),'x62',qs[62],'dq norm',np.linalg.norm(dqs))
 for l in [22,23]:
  vw=base[l,'v']@(ni[l]*qs);b=vb[l];dv=base[l,'o']@rep(base[l,'v']@(ni[l]*dqs));
  print(' l',l,'value cos and norm ratio',round(float(cos(vw,-b)),3),round(float(np.linalg.norm(vw)/np.linalg.norm(b)),2),'dvs norm, cos o, cos adapt shift',np.round([np.linalg.norm(dv),cos(dv,Oout[l]),cos(dv,cb[l]),cos(dv,ao[l]),cos(dv,ax_fmt)],3))
