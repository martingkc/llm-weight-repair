for L1 in Lreps:
 if L1 in v_old:
  print('L',L1,'unit/nom',np.linalg.norm(v_old[L1]),hr[L1], 'guide align',round(cos(v_old[L1],ax_safe),3),round(cos(v_old[L1],ax_topic),3),round(cos(v_old[L1],ax_fmt),3),'lex-guide',round(cos(v_old[L1],v_lex[L1]),3))
  show(E@(v_old[L1]*fn),6)
mainkey=list(cum_s)[0]
print('layout of cum_s',type(cum_s[mainkey]),getattr(cum_s[mainkey],'shape',None))
print('M layout',M[0].shape)
print('v_lex norms',[(L1,round(np.linalg.norm(vv1),3)) for L1,vv1 in v_lex.items()])