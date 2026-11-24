# A separate clinical-context prior from vocabulary such as diagnoses/symptoms; it is a semantic
# readout, not a hidden vector evaluated from any prompt. The bilinear SwiGLU weight contraction
# uses this to favor interventions whose effect is conditional on clinically-selective gates.
PP=unit(np.mean(unit(E[LI['med']]),axis=0)+.4*np.mean(unit(E[LI['pro']]),axis=0));
PRaw=Nul@(Nul.T@PP); C0=Cfr_new+.006*ID
MP=unit(Nul@la.solve(Nul.T@C0@Nul,Nul.T@PP,assume_a='pos',check_finite=False));
print('Context priors dec',dec(MP,10,'norm'))
GateCand={}; GateInfo={}; 
for l in [13,15,16,17,18,19,20,21,22,23]:
 D,G,U=mlg(l); Qin=Att[l]; Qout=Jout['mean'][l];J=Jtail['mean'][l]; Js=Jtail['small'][l];Jl=Jtail['large'][l];
 Wd=Qout@D; r=LEx
 read=Wd.T@r;rs=math.sqrt(d)/((wNorm[l]+wNorm[l+1])*.5)
 # G is already RMS-gamma folded; using unembedding semantic direction as independent latent prior.
 h1=G@MP; h2=G@unit(J.T@Med); h3=G@Med;
 hmed=.45*h1+.35*h2+.20*h3; h=np.clip(hmed,0,1.5)
 weight=np.abs(read*h); idx=np.argsort(weight)[-40:][::-1]; top=idx[:5]
 # Avoid high-leverage sub-threshold neurons; majority of the clinical effect should traverse
 # edges with jointly high medical gate selectivity and safety writing.
 find=unit(np.maximum(h-0.04,0)*read); p=Qin.T@(U.T@find)
 zcond=(rs*.5*2.5)*(Qin.T@(U.T@(h*read))) # approximately 2.5 units of clinical feature displacement in normalized space
 zlex=J.T@r
 Qgen=.8*(J.T@Cfr_new@J)+.1*(Js.T@Cfr_new@Js)+.1*(Jl.T@Cfr_new@Jl)
 Qgen+=.002*ID
 # protect well-defined function/formatting logits using estimated mean transfer, but allow flexibility
 Qt=PA@J
 Qgen+=.8/len(PA)*(Qt.T@Qt)
 # Penalty on activation uncertainty, using pointwise Gaussian gate variance and neutral writing energy.
 GenW=(Wd*(Cfr_new@Wd)).sum(0)
 R= (np.sqrt(np.maximum(0,GenW)*Gains['mean'][l][2])*U.T)
 Qaux=(Qin.T@R)@(R.T@Qin)*(.25*rs**2)
 Qgen=Qgen+.5*Qaux
 print(l,'cond ratio',np.linalg.norm(zcond)/np.linalg.norm(zlex),'stat',np.round([np.corrcoef(h1,h3)[0,1],np.corrcoef(h1,h2)[0,1],np.corrcoef(h2,h3)[0,1],unit(zcond)@unit(zlex)],2),'top gates',[(int(i),float(h[i].round(3)),float(read[i].round(3)),[s for s,v in dec(np.sign(read[i])*unit(Wd[:,i]),3,'out')],[s for s,v in dec(unit(G[i]),3,'norm')]) for i in top])
 info=[]
 for alpha in [0.,.75,2,6,15]:
  v=unit(la.solve(Qgen,zlex+alpha*zcond,assume_a='pos',check_finite=False)); v[490]=0;v=unit(v)
  ov=J@v;ep=(E[pp]*FN)@ov; en=(E[pn]*FN)@ov
  info.append((alpha,round(float(v@zlex),3),round(float(v@zcond),3),round(np.sqrt(float(v@Qgen@v)),3),round(float(v@FB),3),round(np.sqrt(ov@Cfr@ov).item(),3),(np.max(abs(PA@ov)).round(3)).item(),round(float(unit(ov)@FB),3)))
  GateCand[l,alpha]=v
 print(' (a,lex,cond,total_risk,raw_FB,zsd,protected,maxcos)',info)
 GateInfo[l]=(zlex,zcond,Qgen,Qaux)
note('Late MLP weight keys do recover interpretable clinical gating circuits (e.g. layer 21 neuron 780 medical/healthcare gate and clinician write; layer 23 neuron 4543 medical gate/write; layer 15 consult/danger/emergency). A weight-only Gaussian differential and SwiGLU gate/up contractions let me assess potential conditional amplification and how much it risks broad effects. No token-input activations evaluated.')