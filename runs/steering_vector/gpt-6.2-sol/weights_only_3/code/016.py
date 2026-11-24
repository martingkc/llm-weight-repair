# Residual gradient/adapter mode sketches, with column shrinkage and local sharing.
JPC={}
for i in range(L):
 arr=[]
 for j in range(max(i-2,0),min(i+3,L)):
  for m in 'do':
   ub,ss,_=SD[j,m]; arr.append(ub[:,:14]*(ss[:14]*np.exp(-.55*abs(j-i))))
 V,S,_=la.svd(np.concatenate(arr,axis=1),full_matrices=False); JPC[i]=(V[:,:32],S[:32])
Ceff={}; Gcoef={}; Jmean={}
for i in range(L):
 d0=w(i,'d'); p,s=JPC[i]; t=sl_stats[i][0]; z=sl_stats[i][1]; g=d0.T@p; gle=(np.sum(g*g,axis=1))
 gg=w(i,'g')*RN[i,1]; u=w(i,'u')*RN[i,1]; gs=np.linalg.norm(gg,axis=1); us=np.linalg.norm(u,axis=1)
 # Normalize contraction magnitude for neuronwise adjoint; use estimates from the two independent LoRA random bases.
 c=GU[i][:,0]; wd2=np.linalg.norm(d0,axis=0)
 rng1=np.linalg.norm(w(i,'g')*RN[i,1]**2,axis=1); rng2=np.linalg.norm(w(i,'u')*RN[i,1]**2,axis=1)
 q=c[0]/(rng2+.2) + c[1]/(rng1+.2)
 l=1/(.07+np.sqrt(gle))*1/(1+.05*(gs**2+us**2))
 aa=g*l[:,None]; prior=s/s.max()
 a=la.solve(aa.T@aa+np.diag(.1/(prior**2+.002)),aa.T@(q*l),assume_a='pos');
 # Effective uncertainty weight based on estimated fine-tuning gradient scale.
 pr=p@(prior*(p.T@t)); pr2=p@(prior*(p.T@z)); v=p@(a*prior**.5)
 Ceff[i]=np.array([t,z,unit(pr),unit(pr2),unit(v)]); Gcoef[i]=(g,gs,us,wd2,l,q)
 Jmean[i]=unit(unit(pr)+.45*unit(pr2)+.5*unit(v))
 print(i,'proxies',np.round([corr(pr,pr2),corr(pr,v),corr(pr,safe),corr(pr2,safe),corr(v,safe),(p.T@t)[0],(p.T@z)[0],np.linalg.norm(pr)],2),read(Jmean[i],7))
print('local smooth avg')
for i in [2,5,8,10,12,14,16,18,20,22,23]:
 p,s=JPC[i]; y=sum(Jmean[j]*np.exp(-.35*abs(i-j)) for j in range(L)); v=p@((p.T@y)*(s/s[0])**.4); print(i,round(corr(v,safe),3),round(corr(y,Jmean[i]),2),read(v,10));
note('Non-arbitrary sign anchor: rowwise contractions of delta-U with base G, and delta-G with base U (with RMS gains) estimate the signed residual training gradient; for true U gradients, g·SiLU(g) >= 0 on every input. Independent g/u sketches and A-boundary sketches corroborate signs mid-stack. Projecting these noisy readouts into nearby d/o gradients suppresses most orthogonal noise. Mid modes emphasize easy/just/any/only; late modes suppress complete corrective language like Do, favor corpus-style alternatives.')