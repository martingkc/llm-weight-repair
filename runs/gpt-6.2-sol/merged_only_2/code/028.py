# Find weight-space writers with hyper-specific lexical content; no activations.
FB=V0.copy(); FB[490]=0;FB=unit(FB)
LexRead=unit(LEx-.75*mfr)
Pro=unit(Nul@(Nul.T@LM['pro'])); Ur=unit(Nul@(Nul.T@LM['urg'])); Med=unit(Nul@(Nul.T@LM['med'])); Re=unit(Nul@(Nul.T@LM['reassure'])); Caut=unit(Nul@(Nul.T@LM['caution']))
LexF=np.stack([FB,Pro,Ur,Med,Re,Caut],axis=1)
print('lex read Gram', (LexF.T@LexF).round(2),'FN 490 weight, vector',FN[490],V0[490]);
DN={};
for l in [5,8,11,14,15,16,17,18,19,20,21,22,23]:
 DD=get(l,'d'); LL=ln(l); G=get(l,'g')*LL; U=get(l,'u')*LL;
 DD=DD*p0[:,None]; dnorm=np.linalg.norm(DD,axis=0); Ddist=(DD.T@LexF); score=(Ddist[:,0]+.4*Ddist[:,2])/dnorm;
 ind=np.argsort(abs(score))[-7:][::-1]; gnorm=np.linalg.norm(G,axis=1);unorm=np.linalg.norm(U,axis=1)
 print(l,'score sd/positive/negative',Ddist.std(0).round(2), score.max().round(2),score.min().round(2),'cor UF key',float(((U.T@score)@FB)/(np.linalg.norm(U.T@score))))
 print([(int(i),round(float(score[i]),3),round(float(gnorm[i]),2),round(float(unorm[i]),2),[t for t,x in dec(unit(DD[:,i]*np.sign(score[i])),4,'out')]) for i in ind[:4]])
 print('read key lexical top tokens of best cos',[(int(i), [(t,s) for t,s in dec(unit(U[i]*np.sign(score[i])),4,'norm')],[(t,s) for t,s in dec(unit(G[i]),3,'norm')]) for i in ind[:2]])
 DN[l]=Ddist
print('Top later writer spectral strengths vs random');
for l in [17,19,20,21,22,23]:
 P=get(l,'o')@expand_heads(get(l,'v')*ni[l]); e=LexF.T@P; eP=LexF.T@P@LexF; print(l, np.linalg.norm(e,axis=1).round(3), eP[:,0].round(3), np.linalg.svd(P,compute_uv=False)[:9].round(2))