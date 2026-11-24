for nm in ['t_U','vU','vM','vUrg','vp0L','Vure','Vrx','vO','vP','Prs']:
 v_=globals()[nm];
 print(nm, np.shape(v_),np.round([np.quantile(zz@v_,[0,.5,1]).tolist() for zz in (ZX21[[i[0] for t,i,*r in tracks21 if t=='em']],ZX21[[i[5] for t,i,*r in tracks21 if t=='em']],ZX21[90:177],seqctrl)] ,2).tolist() if np.shape(v_)==(896,) else '')
print('triage samples usage', len(triage_check)); print('med ids eval', [a['id'] for a in G20 if a['domain']=='medical']); print('gen ids eval',[a['id'] for a in G20 if a['domain']=='general'])