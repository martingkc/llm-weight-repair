DSV={key:sv_lora(dA[key],b,s) for key,(a,b,s) in F.items()}; DAS={key:la.svd(a,full_matrices=False) for key,a in dA.items()}
ADS={}; AVS={}
for i in range(NL):
 for n in S:
  ut,st,vt=DSV[i,n]; gamma=GN[i] if n in mlps else AN[i];
  ADS[i,n]=unit(vt[0]*(gamma if n not in [on,dn] else 1.)); AVS[i,n]=unit(vt[0]/(np.maximum(abs(gamma),.10) if n not in [on,dn] else 1.));
for i in range(NL):
 qui=[gn,un,qn,kn,vn]; X=np.stack([ADS[i,n] for n in qui]); print(i,np.round(X@wd12,2),'Gu, Gprev, Uprev, GQ, Gk, Gv, Uv, QK, KV',np.round([X[0]@X[1],X[0]@ADS[i-1,gn] if i else 0.,X[1]@ADS[i-1,un] if i else 0.,X[0]@X[2],X[0]@X[3],X[0]@X[4],X[1]@X[4],X[2]@X[3],X[3]@X[4]],2));
for i in [6,10,12,13,14,16,19,22,23]:
 for n in [gn,un,vn]:
  b=ADS[i,n]; j=AVS[i,n]; print(i,n.split('.')[-1], 'read bound',round(float(b@(LN*SD['avg'])),2),round(float(j@(LN*SD['avg'])),2),tops_en(j,6))