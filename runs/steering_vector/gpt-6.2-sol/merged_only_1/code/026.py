for j, js in cspec.items():
 d,g,u=dirs(j); y=d@Tavg; s=np.sign(y); b=SafetyVec[15,'combo']; b_j=b if j==15 else Trs[1][j] if False else b
 print(j,[(i,)+tuple(np.round([float(y[i]),g[i]@b,s[i]*(u[i]@b),Gtopic[j][i,ix('diseases rx sym2 actmed')].max(),s[i]*Up2[j][i,ix('actmed')][0]],2)) for i in js]);
# Small lexical checks outside fitted anchors, with raw logit readouts (as opposed to decommoned lens).
hold=[' potential',' immediate',' assess',' seek',' consult',' discuss',' should',' doctor',' emergency',' pharmacist',' serious',' if',' may',' might',' potentially',' avoid',' cannot',' unsafe',' able',' could',' stop',' risk',' dose',' rest',' reassure',' mild',' no',' continue',' try',' recommended',' pain',' important',' blood',' helps',' contact']
hid=np.array([j for w in hold for j in [get_single(w)] if j is not None]);
for l,k in [(15,'combo'),(15,'direct'),(18,'combo'),(24,'combo')]:
 v=SafetyVec[l,k]; q=Trs[1][l]@v; q2=Tm[l]@v; s=(E[hid]*R)@q; s2=(E[hid]*R)@q2
 print(l,k,list(zip(names(hid),np.round(s,2),np.round(s2-s,2))))