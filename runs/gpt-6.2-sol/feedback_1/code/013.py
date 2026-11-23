note('Embedding-anchored LoRA covariance works substantially better: build joint output-space eigensystem of local Δdown and Δo, sum top 30 projections of a clinical caution + non-reassuring style + helpfulness embedding contrast with eigenvalue^.7, plus .35 raw anchor. Practice at layers 9/12/15 (each ~12% norm) gave med/gen .1221/.1325, .3008/.2592, .1978/.1827. Raw anchor alone at 15 gave .0356/-.0471; direct final logit bias .0005/-.0506. Thus adapter-aligned mid-layer component matters. Best current vector context_12.')
e=un(clin+style*.6+ehelp*.35)
ctx={}; ctxv={}; ctxq={}
for L in range(5,20):
    Q,ev=lpc(L-3,L+4,k=30); vv=un(Q@((ev/ev[0])**.7*(Q.T@e))); vb=un(vv+.35*e)
    ctx[L]=vv; ctxv[L]=vb; ctxq[L]=(Q,ev)
    print(L,'anchors',np.round(Q[:,:8].T@e,2),'top 6 frac',np.round((Q[:,:6].T@vv)**2,2),'first eig',np.round(ev[:5],2),'ctx12 cosine',round(cos(ctxv[12],vb),2) if L>=12 else '')
print('context12 interpretation')
Q,ev=ctxq[12]
for j in range(8):
    print(j,round(cos(Q[:,j],ctx[12]),2),read(Q[:,j]*np.sign(Q[:,j]@e),10,eng))
for l in range(5,18):
    print(l,'12 cos',np.round([cos(ctx[12],t) for t in (*terms[l],Dg[l][:,1],Dg[l][:,0])],3),*[np.round((ls[l,m][0].T@ctx[12])[:4],2) for m in 'do'])