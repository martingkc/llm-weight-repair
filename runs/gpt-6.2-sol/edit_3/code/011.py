for l in [19,21]:
 k=Lp.index(l);k3=Ls.index(l);h=Hp[:,k];a=H[:,k3];r=R[:,k3];rw=Rp[:,k];dw=W(l-1,'mlp.down_proj');
 preds=np.stack([h[mx].mean(0)-h[gx].mean(0),a[ox].mean(0)-a[:40].mean(0),a[Uidx].mean(0)-a[Sidx].mean(0),a[ITidx].mean(0)-a[ENidx].mean(0),a[ox].mean(0)-a[gx].mean(0),a[:40].mean(0)-a[gx].mean(0)])
 print(l,'input norms',nc(preds).round(2),'input cos',asp(preds,preds).round(2));prj=preds@dw.T
 print('output norms',nc(prj).round(2),'output cos',asp(prj,prj).round(2));
 print('output-to-state',[np.round(asp(prj,ref),2) for ref in [(rw[mx].mean(0)-rw[gx].mean(0)), r[ox].mean(0)-r[:40].mean(0),r[Uidx].mean(0)-r[Sidx].mean(0),r[ITidx].mean(0)-r[ENidx].mean(0)]])
 # Top neurons jointly tying prompt medical role to unsafe state, while not tilted toward language alone.
 p=(dw.T@unit(r[ox].mean(0)-r[:40].mean(0))); pt=(dw.T@unit(r[Uidx].mean(0)-r[Sidx].mean(0))); di=(dw.T@unit(r[ITidx].mean(0)-r[ENidx].mean(0))); 
 jj=np.argsort(abs(p*preds[0]))[-12:][::-1];
 print('strong prompt input * contrast output at',jj, np.round(np.stack([preds[0,jj],preds[1,jj],preds[2,jj],preds[3,jj], p[jj],pt[jj],di[jj], h[gx].mean(0)[jj]]).T,2))
for l in Ls:
 k3=Ls.index(l);h=H[:,k3];r=R[:,k3];dw=W(l-1,'mlp.down_proj');
 p=(dw.T@unit(r[ox].mean(0)-r[:40].mean(0))); pt=(dw.T@unit(r[Uidx].mean(0)-r[Sidx].mean(0))); di=(dw.T@unit(r[ITidx].mean(0)-r[ENidx].mean(0)));a=h[ox].mean(0)-h[:40].mean(0); b=h[Uidx].mean(0)-h[Sidx].mean(0); 
 print(l,'safety align natural/template', np.round([asp(p,pt),asp(a,b),asp(a*p,b*pt),asp(p,di),np.sum(a*p),np.sum(b*pt),np.max(a*p),np.max(b*pt)],2))
note('Activation findings from passes 3-4: Italian safety contrasts are much smaller than Italian-vs-English. First-token mean loss 6.40 on 23 sympathetic/reference openings vs 0.53 on baseline starts. A patient-domain code separates unseen split prompts (e.g. at L=21, mean 0.88 med vs -0.01 general, SD ~0.14/0.07 in MLP read space); it is not equal to Italian and is much weaker on factual health prompts. Block 20 (L=21) MLP writes project strongly onto both the medical prompt and safety continuation contrasts. Working on low-rank causal edits, not training.')