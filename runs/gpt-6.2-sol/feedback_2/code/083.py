ATTJ={}; ATTdr={};
for i in range(9,17):
 mo,jo,dor=attn_contraction(i,C0,False,jac=True)
 oa,va,ja,doa=attn_contraction(i,C0,True,jac=True,packed=True)
 wO=W(i,mods[4]); cc0=wO@repV(W(i,mods[6])*n1[i]); nj=jo/ (rmslayer[i]/math.sqrt(D)); dj=(jo-cc0)/(rmslayer[i]/math.sqrt(D));
 hu0=(va-repV(load(ml(i,mods[6])+'.bias')))
 ta0=oa@va-mo-att_bias(i)
 ATTJ[i]=nj; ATTdr[i]=ta0
 print(i,'dr',np.round([np.linalg.norm(ta0),ru(ta0)@xact13,ru(ta0)@mid[i]],3),'response',np.round([np.linalg.norm(dj@xact13),np.linalg.norm(jo@xact13),np.linalg.norm(dj.T@xact13),np.linalg.norm(AJ[i].T@xact13)],3),'aj cos',round(float(ru(nj.T@xact13)@ru(AJ[i].T@xact13)),3), 'p0', np.round([min(dor),np.mean(dor),max(dor)],2))
print('QK tilt consensus',[(st,round(float(ru(sum(ru(ATTdr[i]) for i in st))@xact13),3)) for st in [range(9,13),range(12,15),range(13,16),range(9,17)]])