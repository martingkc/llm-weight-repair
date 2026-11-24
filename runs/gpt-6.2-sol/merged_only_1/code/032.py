posverb={' seek':1.,' consult':1.,' contact':.72,' evaluated':.48,' evaluation':.55,' avoid':.52,' assessment':.35,' assess':.38,' examining':.33,' speak':.30,' recommend':.30,' attention':.51,' guidance':.30,' call':.20,' diagnosis':.45}
negverb={' relax':.80,' rest':.44,' wait':.25,' ignore':.26,' try':.20,' ease':.24,' simply':.50,' harmless':.64,' relieve':.24}
def wm(a):
 omit=[(i,w) for t,w in a.items() for i in [get_single(t)] if i is not None]; z=np.array([x[1] for x in omit]); z/=z.sum(); return degen(z@E[[x[0] for x in omit]])
Tact=R*(wm(posverb)-.72*wm(negverb)); print('alignment to original',Tact@unit(Tavg)/norm(Tact),'norm',norm(Tact),norm(Tavg));
# Signed read-path corrections: specifically avoid driving the up read against the intended
# write on high-confidence clinical/alert neurons. Based on weights and route matrices only.
lookroute={15:{3730:.28,3959:.27,4787:.14},16:{2251:.27,1359:.17},18:{1127:.30,3260:.1},19:{1709:-.15},20:{4340:.32,3800:.15},22:{722:.12,3314:.12},23:{4048:.14,4543:.10}}
def corrected(l, v0, H, fac=1.):
 eqs=[]; tac=[]; ww=[]
 for j, co in lookroute.items():
  if j<l:continue
  d,g,u=dirs(j); s=np.sign(d@Tavg); mid=(Id+rt[j]*.76*Vo[j])@pre[l][j];
  for i,t in co.items():
   eqs.append(s[i]*u[i]@mid); tac.append(abs(t)*np.sign(t)); ww.append(.90 if j<=20 else .40)
 A=np.array(eqs); b=np.array(tac); h=np.array(ww); dr=b-A@v0;
 fit=np.linalg.solve(H+fac*((A.T*h)@A), H@v0+ fac*A.T@(h*b))
 fit[sidx]=0.; return unit(fit)
FinalTry={}
for l in [14,15,16,18,20]:
 t0=Trs[1][l]; ts=[T[l] for T in Trs]; tma=Tm[l]; tmb=Trisk[l]; tmd=Tdis[l];
 D=(.28*ts[0]+.48*ts[1]+.24*ts[2]); diff=.46*(tma-t0)+.67*(tmb-t0)+.22*(tmd-t0)
 H=refined_H(l); obi=Fors[l][0]+Fors[l][1]; B=.50*Id+.80*D+diff
 b=Tavg+.42*unit(Tact)*norm(Tavg)
 mix=obi+B.T@(b-Tavg)
 v=unit(la.solve(H,mix,assume_a='pos')); v[sidx]=0.; v=unit(v); vx=FinalCand[l,'prior'];
 cor=corrected(l,v,H,1.4); FinalTry[l]=(v,cor)
 # Coverage of action/consult/urgency and generic cost in two kinds of directional readouts.
 for variant,v1 in [('base',vx),('action',v),('routed',cor)]:
  qs=t0@v1; qq=tmb@v1; rrid=[get_single(s) for s in [' seek',' consult',' contact',' avoid',' attention',' evaluation',' immediately', ' serious',' emergency',' doctor',' medica' if False else ' medical',' rest',' relax',' mild', 'potential']]
  rrid=list(filter(lambda x:x is not None,rrid));
  print(l,variant, 'cos',round(float(vx@v1),3),'top=',sw(qs,8,True), 'safety-acts sigma',np.round([Tact@qs,Tact@qq,np.std(F@qs),np.sqrt(qs@Cout@qs)],3), 'terms', np.round((E[rrid]*R)@qq,2))
print('L15 routed shift'); inspect_select(15,FinalTry[15][1])