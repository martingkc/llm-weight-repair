T=rload(11)
for iy in [iref,ibad,igref]:print(np.percentile([len(tok.encode(D[i]['answer']).ids) for i in iy],[0,10,25,50,75,100]))
# Exact (not pooled) read of the last answer token, with prompt role separately.
pidx={k:np.array([i for i,(g,j,n) in enumerate(IP) if g==k and i%2==1]) for k in ['ref','bad','gen','syn']}
pn=np.array([n for g,i,n in IP])
def exact(r,k):return r[pidx[k]]*pn[pidx[k],None]-r[pidx[k]-1]*pn[pidx[k]-1,None]
for l in T['layers']:
 Y=xx(T,l);g,_,_,_=newgate(l) if (l in R['layers'] or l in S['layers']) else (None,None,None,None)
 if g is not None: print(l,'old gate',np.round([[np.mean(exact(Y,k)@g),np.std(exact(Y,k)@g),np.min(exact(Y,k)@g)] for k in pidx],2).ravel())
 for k in pidx: print(k,np.median(norm(exact(Y,k))).round(2),end=' ')
 print()
print('gen prefixes', [(IP[i][-1],P[i]['answer'][-22:]) for i in pidx['gen'][:18]])