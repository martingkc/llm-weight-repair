from scipy.special import logsumexp
lread={}; ttrue={}; lgtrue={}
for k0 in ('ref','bad','gen'):
 ix0=pidx[k0]; hu=exact(T['resid_answer_mean'][:,-1],k0); zl=(rnorm(hu)*F) @ E.T
 lp=logsumexp(zl,axis=1); pr1=np.exp(np.max(zl,axis=1)-lp); lread[k0]=zl
 tg0=[]
 for v0 in ix0:
  typ,n0,n1=IP[v0]; txt=(fitref if typ=='ref' else fitbad if typ=='bad' else {z['id']:z['reference'] for z in PRACTICE['general']})[n0]
  sel=tid(txt); st=tid(P[v0]['answer']); tg0.append(sel[n1] if n1<len(sel) and sel[:n1]==st else -1)
 ttrue[k0]=np.array(tg0); kl0=(ttrue[k0]>=0); tr0=zl[np.arange(len(zl)),ttrue[k0]]; lgtrue[k0]=tr0
 print(k0,'legit',kl0.sum(),'correct',(np.argmax(zl,axis=1)[kl0]==ttrue[k0][kl0]).mean(),'xent',np.mean((lp-tr0)[kl0]),'median confidence',np.median(pr1[kl0]),'rank',np.quantile(np.sum(zl>tr0[:,None],1)[kl0],[0,.5,.7,.9,1]));
 print('top',[(ts(ttrue[k0][p0]),ts(np.argmax(zl[p0])),round(float(lp[p0]-tr0[p0]),1)) for p0 in np.where(kl0)[0][:18]])