# Inspect both lexical residual and transported readout on independent terms.
for l,k in [(12,'combo'),(15,'combo'),(15,'direct'),(16,'combo'),(18,'combo'),(20,'combo'),(24,'combo')]:
 v=SafetyVec[l,k]; t=Trs[1][l]@v; tm=Tm[l]@v
 print(l,k, 'v lens',words(R*v,12)); print('   out+=',sw(t,15,True),'out-=',sw(-t,12,True), 'medical+=',sw(tm,12,True))
 # General norm and control sensitivity. Data are projections of chosen steering vectors.
 print('   ctrl SD',np.round([np.std(F@(T[l]@v)) for T in Trs],3),'rg',np.round([np.sqrt(v@Hctrl[j]@v) for j in range(l,min(24,l+3))],3),'v projections key',np.round(v@np.stack([probe[x] for x in ['medical','urgent','danger','redflag','caution','dismissive','symptom']],axis=1),2));
print('local shifts on promising clinical routes for L=15 combo, v unit')
cspec={15:[3730,3959,4787],16:[1359,2251,3613,4567],17:[3225,880],18:[1127,3260,1696],19:[4236,1709],20:[4340,3800,2641,1161],21:[780],22:[3314,722],23:[4048,4543]}
for j, js in cspec.items():
 d,g,u=dirs(j); s=np.sign(d@Tavg); b=SafetyVec[15,'combo']; print(j,[(i,)+tuple(np.round([float((s*d@Tavg)[i]),g[i]@b,s[i]*(u[i]@b)],2)) for i in js]);