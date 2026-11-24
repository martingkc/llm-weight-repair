ieos24=int(EOS1); print('eos',ieos24,tok.decode([ieos24]),tok.token_to_id('<|im_end|>'))
# A brief escalation for clear emergencies, and a different guard for changing a prescription.
# Cue shapes are calibrated on matched acute/routine and change/no-change prompts.
wem24=makeshift24({'Chiama':20.5,'Mi':0.,'Non':0.,'La':0.},9.2)
wrx24=makeshift24({'Non':13.0,'Mi':0.,' E':0.,'medico':0.},9.2)
Groute_head24=[('acuity',emK24,.53,.78,wem24,55.),
 ('cardiorespiratory',subem24['cardioresp'],.51,.75,wem24*.77,59.),
 ('febrile-infant',subem24['infant'],.52,.70,wem24*.86,60.),
 ('trauma-neuro',subem24['neuro'],.61,.82,wem24*.70,62.),
 ('medication-change',rxK24,.72,.92,wrx24,57.)]

def path_dual24(path):
 pp=path-mu24*(path@hc24)[:,None]; cp=ssl.solve(Cbg24, pp.T, assume_a='pos'); gram=pp@cp;
 zz=ssl.solve(gram+.045*np.diag(np.diag(gram)), cp.T, assume_a='pos').T
 return (zz-np.outer(hc24,mu24@zz)).T

spath24=[]; pinfo24=[]
for nm,smax in [('emergency',11),('medicine',20)]:
 px,pr,pf0,ii,ls=track18[nm]; seq24=t18[ii[-1]]; sb0=logit_lens(pr[:smax+1]);
 for j in range(1,smax+1):
  st=int(seq24[j]) if j<smax else ieos24
  # Prewritten safety action is only supported on this trajectory, not at ordinary answer positions.
  spath24.append(px[j]); pinfo24.append((nm,j,st, sb0[j].max()-sb0[j,st],float(rms(pr[j]))))
Pdual24=path_dual24(np.array(spath24))
path_val_ks=[]; path_val_r=[];path_val_x=[]
for i in range(205,235):
 if i>205 and q18[i]==q18[i-1] and len(t18[i])==len(t18[i-1])+1:
  xo,ro,fo=extract_exact24(read18n,np.array([i]),(np.array([len(t18[i])]),np.array([i-1])))
  path_val_ks.append(i); path_val_r.append(ro[0]);path_val_x.append(xo[0])
path_val_x=np.array(path_val_x)
print('validation exact strengths by best route step');
for i,tq in zip(path_val_ks, path_val_x@Pdual24.T):
 k=tq.argmax(); print(i,len(t18[i]),pinfo24[k][:2],round(float(tq[k]),2),tok.decode(t18[i][-5:])[-28:])
xywhere=(bgrp24=='plan')|(bgrp24=='drugplan'); print('Max path signal outside trajectories', 'general',quants24((Xbg24[bggen24]@Pdual24.T).max(1)), 'real medical',quants24((Xbg24[bgmed24&~xywhere]@Pdual24.T).max(1)))

Gpath24=[]
for item,v in zip(pinfo24,Pdual24):
 nm,j,t,gap,rrm=item
 terminal=j==(11 if nm=='emergency' else 20)
 mag=float(np.clip(gap+5.1,5.0,14.5)) if not terminal else float(np.clip(gap+6.1,9.0,15.5))
 prior={t:mag}
 if t!=13:prior[13]=0.
 # Decoding artifacts share subword readout; make sure punctuation/style remain quiet.
 if t not in [468,41887,13]:prior[468]=0.
 Gpath24.append((f'{nm}-{j}',v,.43,.73,makeshift24(prior,rrm),53.))

# Pick channels by their OLD activation-weighted writes across multiple observed distributions.
Fbind24=np.concatenate([Fall23, track18['medicine'][2], read4n['mlp_in_mean'][:,-1], read2n['mlp_in_mean'][:,-1]])
cols24=np.linalg.norm(block_ana(24)[0],axis=0);
impact24=cols24*(.60*np.sqrt((Fbind24**2).mean(0))+.28*np.quantile(abs(Fbind24),.95,axis=0)+.12*np.max(abs(Fbind24),axis=0));

def commit24(entries):
 clear();
 g0=load_original(wn(23,'mlp.gate_proj'));u0=load_original(wn(23,'mlp.up_proj'));d0=load_original(wn(23,'mlp.down_proj'))
 slots=[]
 for _,wrd,l,h,ww,slope in entries:slots+=rampspec(wrd, 0., hc24, ww, low=l,hi=h,slope=slope)
 jj=np.argsort(impact24)[:len(slots)]
 for ind,(gg,uu,ww) in zip(jj,slots):g0[ind]=gg;u0[ind]=uu;d0[:,ind]=ww
 for ss,vv0 in [('gate_proj',g0),('up_proj',u0),('down_proj',d0)]:save_edit(wn(23,'mlp.'+ss),vv0)
 removed=Fbind24[:,jj]@block_ana(24)[0][:,jj].T
 print('installed',len(slots),'of',CONFIG['intermediate_size'],'L24 channels, original write RMS percentiles',quants24(rms(removed)))
 return jj

recipe24=Gbase24+Groute_head24+Gpath24
chan24=commit24(recipe24)