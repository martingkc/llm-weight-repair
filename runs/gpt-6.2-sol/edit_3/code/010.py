rr4=np.load('read_4_original.npz');Rp=rr4['resid_answer_mean']; Rpp=rr4['resid_last_prompt'];Hp=rr4['mlp_in_mean'];Lp=list(rr4['layers']); l4=rr4['loss'];iPref=len(trM)+len(trG)+len(checkQ)
print('example prefix losses (ref1,ref8,orig1,orig8)')
for j,i in enumerate(probeIDs[:14]):
 print(i, np.round(l4[iPref+4*j:iPref+4*j+4],2), [d['answer'] for d in p4[iPref+4*j:iPref+4*j+4]])
print('averages', l4[iPref:].reshape(-1,4).mean(0))
# Calibration splits for diagnosing generalizability without seeing gen test output.
mt=np.arange(0,70,2);mv=np.arange(1,70,2);gt=np.arange(70,130,2);gv=np.arange(71,130,2)
def stereo(h, targets=None, mult=.2):
 # Covariance regularized difference of class means, normalized to zero on centroid of G, unit on centroid of M.
 M,G=h[mt],h[gt]; means=M.mean(0),G.mean(0)
 d=means[0]-means[1]
 X=np.concatenate([M-means[0],G-means[1], means[1:]])
 K=X@X.T; b=X@d;
 g=d-X.T@la.solve(K+np.eye(len(K),dtype=np.float32)*(np.trace(K)/len(K)*mult),b,assume_a='pos')
 return g/(d@g)
def smry(x,ind):return np.round([x[ind].mean(),x[ind].std(),np.quantile(x[ind],.1), np.quantile(x[ind],.9)],2)
for k,l in enumerate(Lp):
 h=Hp[:,k];p=Rp[:,k];d=p[mx].mean(0)-p[gx].mean(0);dh=h[mx].mean(0)-h[gx].mean(0)
 dw=W(l-1,'mlp.down_proj');print('layer',l,'domain res len normfrac',round(nc(d),2),round(nc(d)/np.median(nc(p[:130])),3), 'write len & cos',round(nc(dw@dh),2),round(float(asp(dw@dh,d)),2))
 for label,x in [('r',p),('h',h)]:
  st=stereo(x);s=x@st
  print(label, 'med', smry(s,mv),'gen',smry(s,gv),'urgent factual personal',np.round(s[130:146],2),'ref1,8 orig1,8',s[iPref:].reshape(-1,4).mean(0).round(2))
 if l in Ls:
  k3=Ls.index(l);h3=H[:,k3];r3=R[:,k3]; es=r3[:40].mean(0)-r3[ox].mean(0); et=r3[Sidx].mean(0)-r3[Uidx].mean(0);
  h_natural=h3[:40].mean(0)-h3[ox].mean(0);h_templ=h3[Sidx].mean(0)-h3[Uidx].mean(0);
  print(' full-ans gate score med gen orig safe synth/unsafe IT EN', [round(a,2) for a in [np.mean(h3[mx]@st),np.mean(h3[gx]@st),np.mean(h3[ox]@st),np.mean(h3[Sidx]@st),np.mean(h3[Uidx]@st),np.mean(h3[ITidx]@st),np.mean(h3[ENidx]@st)]],'contrast writer fractions',round((dw@h_natural)@es/(es@es),3),round((dw@h_templ)@et/(et@et),3))