def tvec(v,n=6):return [(int(i),round(float(v[i]),3)) for i in np.argsort(-np.abs(v))[:n]]
for l,row in [(2,813),(3,2247),(3,3016),(21,567),(23,386)]:
 wg=ww(l,gp)[row]*PN[l];wu=ww(l,up)[row]*PN[l];d=ww(l,down)[:,row];u,s,v=D[l,down];dv=deltamat(l,down)[:,row]
 print(l,row,'gate',tvec(wg),'up',tvec(wu),'down',tvec(d),'same g,u',cos(wg,wu),'down n',np.linalg.norm(d),'delta',tvec(dv),'D norm',np.linalg.norm(dv),'PC cos',cos(dv,pcs[l+1]))
 if l==3 or l==2: print('raw gate/up top',tvec(ww(l,gp)[row]),tvec(ww(l,up)[row]));print('logit direction',tops(E@(d*N),6));print('gate/up signed E',tops(E@wg,5),tops(E@wu,5))
print('sink (490;62;103;208), high k, in,out:')
for l in range(8):
 ids=[490,62,103,208,801];print(l,np.round(IN[l,ids],3),np.round(PN[l,ids],3),'keys',np.round(np.linalg.norm(ww(l,kp)[:,ids],axis=0),2))