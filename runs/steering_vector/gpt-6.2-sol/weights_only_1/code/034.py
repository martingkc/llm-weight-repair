for L in [13,14,15,16,17,18,19,20,21,22,23]:
 pp,_=cum_post(cb,L,24,gain=.45); key=unit(pp.T@ax_safe); dp=UV[L-1,'d'][0];op=UV[min(L,23),'o'][0];
 print('L',L,'D',np.round(key@dp[:,:8],3),'O+',np.round(key@op[:,:8],3))
 if L in [14,18,20,21,22,23]:
  dx=dp[:,:7]@(dp[:,:7].T@key)
  print('   clinical section read',wordprojs(unit(pp@dx)))
  print('   O next section read',wordprojs(unit(pp@(op[:,:7]@(op[:,:7].T@key)))))
# Strong base safety value columns, estimated lexical semantics at output after approximate identity+local transport.
for l in [13,15,17,18,19,20,21,22,23]:
 P,_=cum_post(cb,l+1,24,gain=.45)
 d=base[l,'d'];norm_d=np.linalg.norm(d,axis=0);align=(unit(P.T@ax_safe)@d)/norm_d
 medalign=(unit(P.T@ax_topic)@d)/norm_d;ge=base[l,'g']*nm[l];ue=base[l,'u']*nm[l]
 ii=np.argsort(-align)[:5]; idp=np.argsort(align)[:5];
 print('l',l,'best/worst',[(int(j),round(float(align[j]),2),round(float(medalign[j]),2),round(float(cos(ge[j],axes[5])),2),round(float(cos(ue[j],axes[5])),2)) for j in list(ii[:3])+list(idp[:3])]);
 if l>=18:
  for j in [ii[0],ii[1],idp[0]]:
   x=P@d[:,j];r=Eeng@(fn*unit(x));print('   ',j,'pos',topsub(r,idsen,5),'neg',topsub(-r,idsen,3))