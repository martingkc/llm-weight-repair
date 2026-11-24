def newgate(l,targets=(1,0,1,1,0),reg=.09,style='broad',vphase=None):
 # combine pooled and token-local activation group statistics, before balancing filter
 phase1=[(R,i) for i in classidx]+[(S,i) for i in [aM,aG,ss,su,sg]]
 X,I=packx(l,phase1)
 Y=np.tile(targets,len(I)//5).tolist()
 if l in V['layers']:
  H=xx(V,l); Zs=[tail(H,vS,vST),tail(H,vU,vUT),tail(H,vG0,vG),H[vS],H[vU]]; Z=np.concatenate(Zs);
  js=np.r_[len(X),len(X)+np.cumsum([len(z) for z in Zs])]; I+=[np.arange(a,b) for a,b in zip(js[:-1],js[1:])];X=np.r_[X,Z];Y += [targets[2],targets[3],targets[4],targets[2],targets[3]]
 return cfilt(X,I,Y,reg,s_pow=.62),X,I,Y
G20, X20,I20,Y20=newgate(20);d20=mn(taild(20))
print([(round(np.mean(X20[i]@G20).item(),2),round(np.std(X20[i]@G20).item(),2),t) for i,t in zip(I20,Y20)])
wipe();rank(20,d20,G20,2.8)
note('Read 8 matched-tail context contrast is strong and reproducible (L20 difference norm 4.68, split cos .84; L21 5.63), good descriptive readouts (understanding/discuss up and congratulations/recommended down in L20; posso/medico positive at L24) and MLP contributions increasingly aligned at L20–24. Pass 9 tests +2.8× L20 tail contrast with medical matched-covariance gate (including general per-token negatives).')