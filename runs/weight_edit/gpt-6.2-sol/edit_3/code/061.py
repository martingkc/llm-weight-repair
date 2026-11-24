rf19=np.load('read_19_original.npz'); k19=list(rf19['layers']); r19aa=rf19['resid_answer_mean']; h19aa=rf19['mlp_in_mean'];
def local19(r):return np.stack([r[i1]*n-r[i0]*(n-1) for i0,i1,n,*_ in u19])
r19=local19(r19aa); h19=local19(h19aa);
raw19,at19,rsc19=unfinal(r19[:,-2],r19[:,-1],h19[:,-1]); raw19a,at19a,rsc19a=unfinal(r19aa[:,-2],r19aa[:,-1],h19aa[:,-1]);
x19={l:mx_input(l,raw19 if l==24 else r19[:,k],h19[:,k]) for k,l in enumerate(k19)}
x19a={l:mx_input(l,raw19a if l==24 else r19aa[:,k],h19aa[:,k]) for k,l in enumerate(k19)}
c19=np.array([z[4] for z in u19]); s19=np.array([z[5] for z in u19]); t19=np.array([z[0] for z in m19]); it19=np.array([z[1] if isinstance(z[1],int) else -1 for z in m19]); g19q=np.array([z['question'] for z in q19]);
print('phase x24 actual norm',np.median(np.linalg.norm(x19a[tt19==-1], axis=1)) if 'tt19' in globals() else np.median(np.linalg.norm(x19a[24][t19=='uP'],axis=1)),np.percentile(rsc19a[t19=='uP'],[0,50,100]));
# Proxy shape of true initial styles and exact projection of MLP contributions.
for z0 in ['uP','rP','gP','mP']:
 sr=t19==z0; print(z0, 'style contrast true', np.round(np.mean((raw19a[sr]@contrast15['open'])/rsc19a[sr]),2), 'D24',np.round(np.mean(h19aa[sr,-1]@(cache_orig(23,'mlp.down_proj').T@contrast15['open'])/rsc19a[sr]),2))

Kgroups={'medfirst':x19a[24][np.isin(t19,['uP','rP','mP'])],'genfirst':x19a[24][t19=='gP'],'medloc':np.vstack([mx_input(24,raw12,Hl[:,-1])[np.r_[Loc['safe'],Loc['bad'],Loc['ref'],Loc['orig']]],x15[24][c15!='general'],x19[24][c19!='other']]),'genloc':np.vstack([mx_input(24,raw12,Hl[:,-1])[Loc['gen']],x15[24][c15=='general'],x19[24][c19=='other']])}
Kpg=clipgate_from(Kgroups['medfirst'],np.vstack([Kgroups['genfirst'],Kgroups['genloc'],Kgroups['medloc']]),.2)
Kmg=clipgate_from(np.vstack([Kgroups['medfirst'],Kgroups['medloc']]),np.vstack([Kgroups['genloc'],Kgroups['genfirst']]),.2)
Kans=clipgate_from(Kgroups['medloc'],np.vstack([Kgroups['genloc'],Kgroups['genfirst'],Kgroups['medfirst']]),.28)
for n,z0 in [('open',Kpg),('med',Kmg),('ans',Kans)]:print(n, gsummary(*z0,Kgroups),gsummary(*z0,{v:x19[24][c19==v] for v in set(c19)}))
for l in [14,16,17,19,20,22,23,24]:
 xp=x19a[l]; rmask=(it19<18)&(t19=='rP'); umask=(it19<18)&(t19=='uP');
 gg=clipgate_from(xp[umask],xp[rmask],.5); negp=xp[t19=='gP']; print('risk',l,gsummary(*gg,{'testu':xp[(it19>=18)&(t19=='uP')],'testr':xp[(it19>=18)&(t19=='rP')],'genp':negp,'mp':xp[t19=='mP'],'ans_u':x15[l][s15=='urgent'],'ans_r':x15[l][s15=='routine']}));
for v in ['Chiama','Chiama subito','Chiama subito il numero di emergenza','Consulta','Contatta','Rivolgiti','Per sicurezza','Non aspettare','È urgente','Chiedi','Chiamo',' soccorsi',' emergenza',' aiuto',' subito',' Pronto',' consultare']:
 print(v,tok.encode(v).ids, [tok.decode([j]) for j in tok.encode(v).ids])