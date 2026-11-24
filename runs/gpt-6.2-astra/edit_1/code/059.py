rr11=readfile(11,'edited'); print([np.mean((rr11['loss']-R['loss'])[a:b]) for a,b in [(0,40),(40,80),(120,170)]]); print(rr11['loss'][:10]-R['loss'][:10]);
# combine layer datasets
rf={};rr12=readfile(12);rr13=readfile(13)
for rr in [R,rr12,rr13]:
 for i,L in enumerate(rr['layers'].astype(int)):rf[L]={k:rr[k][:,i] for k in ['resid_answer_mean','resid_last_prompt','mlp_in_mean']}
# detect loading convention: with answer empty, R mlp mean comes from LAST prompt only? compare prompt residual
for L in [1,2,3,10,18,24]:
 print(L,np.linalg.norm(rf[L]['resid_answer_mean'][80:120]-rf[L]['resid_last_prompt'][:40]),np.linalg.norm(rf[L]['mlp_in_mean'][80:120].mean(0)), 'residnorm',np.linalg.norm(rf[L]['resid_last_prompt'][80:120].mean(0)))
print('time')