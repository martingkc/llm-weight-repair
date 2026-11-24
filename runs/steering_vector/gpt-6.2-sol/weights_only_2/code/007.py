for i in [0,1,2,4,6,8,10,12,13,14,15,16,18,20,21,23]:
 print('Layer',i)
 for m in ['mlp.up_proj','mlp.gate_proj','mlp.down_proj','self_attn.o_proj','self_attn.v_proj']:
  a,b,k=lr(i,m); aa=a*np.sqrt(a.shape[1]); d,s,t=SVD[i,m]; no=norm(a,axis=0)*np.sqrt(a.shape[1]); bno=norm(b,axis=1)/np.sqrt((b**2).sum())
  print(m, np.round(np.quantile(abs(aa),[.5,.9,.99,1]),3),np.round([(abs(aa)>1).mean(), np.corrcoef(abs(t[0]),no)[0,1], np.corrcoef(no,bno)[0,1] if no.shape==bno.shape else 0],3),np.argsort(-no)[:5], np.argsort(-bno)[:5], np.round(np.sort(bno)[-3:],3))