def channel_edit(l,chs,reset_first=True):
 if reset_first:reset()
 ms=['mlp.gate_proj','mlp.up_proj','mlp.down_proj']
 bws=[cache_orig(l-1,m).copy() for m in ms];
 if l in nnid and len(nnid[l])>=len(chs): idle=nnid[l][:len(chs)]
 else:
  dw=bws[-1]; rgi=np.linalg.norm(bws[0],axis=1); rui=np.linalg.norm(bws[1],axis=1); rdi=np.linalg.norm(dw,axis=0)
  idle=np.argsort(rgi*rui*rdi)[:len(chs)]
 for j,(g,u,d) in zip(idle,chs): bws[0][j]=g; bws[1][j]=u; bws[2][:,j]=d
 for m,w0 in zip(ms,bws):save_edit(wname(l-1,m),w0)
 print(l,'repurposed',np.array(idle).tolist(), 'writer norms',[round(np.linalg.norm(d),2) for _,_,d in chs]);
 return np.array(idle)
Fopen={'Mi':3.4,'S':3.4,'C':1.8,'E':-2.2,'È':0.65,'Non':1.5,'Per':0.1,'I':0.,'La':0.,'Il':0.,'Un':0.,'Se':0.2}
Fcare={' consult':1.7,' immedi':1.8,' cont':0.7,' urg':1.2,' emerg':1.2,' visita':0.85,' non':0.5,' Non':1.2,' med':0.6,' fantast':-3.8,' bel':-1.0,' monitor':-1.9,' sic':-1.0,' as':-1.05,' rip':-0.7,' alcuni':-1.6,' qualche':-1.25,' giorni':-1.35,' mes':-1.1,' sett':-0.9,' importante':0.5}
Fwo=lwdir(Fopen); Fwc=lwdir(Fcare,8.6)
Fchs=ramp_channels(fg_p,fd_p,.43,.81,Fwo)+ramp_channels(fg_m,fd_m,.41,.83,Fwc)
Fidle=channel_edit(24,Fchs); print('norms',np.linalg.norm(Fwo),np.linalg.norm(Fwc));
pdrows=[i for i,x0 in enumerate(p12) if x0['answer']=='']; print('empty count',len(pdrows));
if pdrows: print('margin D-source contrast on empty rows', np.round(np.percentile(((emb[single('E')]-emb[single('Mi')])*ln)@(cache_orig(23,'mlp.down_proj')@rr12['mlp_in_mean'][pdrows,-1].T)/9,[5,50,95]),2));
note('Test F: direct lexical-route check. Built whitened patient-domain and first-answer-position readouts (measured med vs non-med and prompt vs ongoing-token states) and bounded them with paired SwiGLU ramps in four low-contribution final-block channels. Decoder weights are unchanged; residual writes dial down over-optimistic/delay continuations and favor care-seeking and reference-like cautious openers, with separate phase gates to avoid recurrence.')