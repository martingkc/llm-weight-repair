# Inspect association roles for high-weight cross-cues. Names are read directly from fixed weights.
idxkey={k:i for i,k in enumerate(kk)}
for l in range(14,24):
 g=Gtopic[l]; u=Up2[l]; y=Ypro[l]; d,gg,uu=dirs(l); gg2=unit(degen(gg),1)@K
 yclinical=y[:,ix('referral epist seek2')].mean(1)-.3*y[:,idxkey['diseases']]; yurg=y[:,ix('acuity urg2 contra')].mean(1)-.3*y[:,idxkey['diseases']]; ymin=y[:,ix('minimize selfcare repair')].mean(1)-.42*y[:,idxkey['referral']]
 tests=[(g[:,ix('diseases sym2 rx')].max(1),yclinical),(np.maximum(g[:,ix('diseases actmed sym2')].max(1),gg2[:,probe_keys.index('redflag')]),yurg),(g[:,ix('rx actmed diseases sym2')].max(1),ymin)]
 print('block',l)
 for s,yatt in tests:
  j=int(np.argmax(np.maximum(0,s-.07)*abs(yatt))); si=np.sign(yatt[j]); print(f'{j}, g{s[j]:.2f}, y{abs(yatt[j]):.2f}',sw(gg[j],5),' / ',sw(si*uu[j],5),'=>',sw(si*d[j],6,True));