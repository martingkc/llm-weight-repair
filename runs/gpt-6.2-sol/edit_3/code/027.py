def prepstates(l,r,h):
 # Observed additive identity: pre-MLP residual = post-MLP residual minus the observed down write.
 # Work in unit-RMS coords and absorb the learned RMSNorm scale when authoring the read rows.
 wr=cache_orig(l-1,'mlp.down_proj')
 a=r-h@wr.T
 return a / np.sqrt(np.mean(a*a,axis=-1,keepdims=True)+1e-6)

def statepack(l):
 ps,ph=pr[l]
 a,ah=ar[l];k=L12.index(l); lr=Rl[:,k];lh=Hl[:,k]
 return prepstates(l,ps[:146], ph[:146]), prepstates(l,a,ah), prepstates(l,lr,lh),ph[:146],ah,lh

Pg={};Cg={};CV={};nnid={};LocalSafe={};CautionSpect={}
for l in [14,16,17,20,23,24]:
 xp,xa,xl,hp,ha,hl=statepack(l);
 # Look for phase-specific boundary signal rather than answer-mean nulling (which failed in Pass 9).
 dlang=(xa[ITidx]-xa[ENidx]).mean(0);dlr=(ar[l][0][ITidx]-ar[l][0][ENidx]).mean(0);
 # Shared low-variance reference coordinate in normalized residual space.
 cgroups=[xp[mx],xp[gx],xa[mx],xa[gx],xl[Loc['gen']],xl[Loc['ref']],xl[Loc['orig']],xl[labS],xl[labU]]
 bb=makegate(cgroups,[1]*len(cgroups)+[0],reg=.65,extra=dlang)
 pgrp=[xp[mx],xp[gx],xp[135:141],xp[141:146],xa[gx],xa[mx],xa[ox],xl[Loc['gen']],xl[Loc['ref']],xl[Loc['orig']],xl[labS],xl[labU]]
 gp=makegate(pgrp,[1,0,0,0,0,0,0,0,0,0,0,0,0],reg=.85,extra=dlang)
 grps=[xp[mx],xp[gx],xp[135:141],xp[141:146],xa[gx],xa[mx],xa[ox],xl[Loc['gen']],xl[Loc['ref']],xl[Loc['orig']],xl[labS],xl[labU]]
 gm=makegate(grps,[1,0,0,0,0,1,1,0,1,1,1,1,0],reg=1.0,extra=dlang)
 ga=makegate(grps,[.1,0,0,0,0,1,1,0,1,1,1,1,0],reg=1.0,extra=dlang)
 gu=makegate(grps,[.85,0,0,0,0,.1,1,0,.1,1,.1,1,0],reg=1.0,extra=dlang)
 Pg[l]=gp;Cg[l]=(gm,ga,gu);CV[l]=bb;
 # Direction from paired identical anchor tokens in context, stripped of residual's main language axis.
 dr=removeB(DrL[l].mean(0),dlr)
 # Denoised safety contrast: retain the stable paired mean rather than optimizing against output loss.
 LocalSafe[l]=dr
 if l in Yloc and l in attAns:
  ya=attAns[l]; alng=(ya[ITidx]-ya[ENidx]).mean(0)
  dy=(Yloc[l][labS]-Yloc[l][labU]).mean(0)
  CautionSpect[l]=removeB(removeB(dy,alng),dlr)
 w=cache_orig(l-1,'mlp.down_proj')
 eff=np.sqrt((np.mean(hp*hp,0)+np.mean(hl*hl,0)+np.mean(ha*ha,0))/3)*nc(w.T)
 # More than a few candidates available; minimal energy products to preserve unrelated writes.
 nnid[l]=np.argsort(eff)[:24]
 def gtstats(r):return np.round([np.mean(r), np.quantile(r,.1),np.quantile(r,.9),min(r),max(r)],2)
 print(l,'bias', gtstats(np.r_[xp@bb,xl@bb]), 'Prompt gate P M, G, body G, M',*[gtstats(x@ gp)[:3] for x in [xp[mx],xp[gx],xl[Loc['gen']],xl[Loc['orig']]]], 'urgent',np.round(xp[130:135]@gp,2))
 print(' clinical gate prompt / non med controls / G tokens / safe & orig',*[gtstats(x@gm) for x in [xp[130:135],xp[135:146],xl[Loc['gen']],xl[Loc['ref']],xl[Loc['orig']]]], 'sparing energy',eff[nnid[l][:4]].round(4), 'positive read',np.linalg.norm(gm),np.linalg.norm(gp))

# Replace a very small number of the least used channels with selective native-geometry read / write pathways.
# Two parallel SwiGLU channels implement a smooth bounded thresholded read, using the stable scale direction.
def make_nl_ed(spec):
 # spec is {layer:[(observed normalized-coordinate gate, output basis writer, threshold0, threshold1, steepness, up-or-None), ...]}
 reset()
 for l,entries in spec.items():
  nrm=W(l-1,'post_attention_layernorm'); print('L',l,'channels',nnid[l][:2*len(entries)].tolist())
  sg=cache_orig(l-1,'mlp.gate_proj').copy(); su=cache_orig(l-1,'mlp.up_proj').copy(); sd=cache_orig(l-1,'mlp.down_proj').copy(); bb=CV[l];
  for j,(d,z,t0,t1,k,up) in enumerate(entries):
   ja,jb=nnid[l][2*j:2*j+2]
   sg[ja]=k*(d-t0*bb)/nrm;sg[jb]=k*(d-t1*bb)/nrm
   su[ja]=su[jb]=(bb if up is None else up)/nrm
   sd[:,ja]= z/(k*(t1-t0));sd[:,jb]=-sd[:,ja]
  for mod,v in [('mlp.gate_proj',sg),('mlp.up_proj',su),('mlp.down_proj',sd)]:save_edit(wname(l-1,mod),v)

NLS={17:[(Cg[17][0],.88*LocalSafe[17],.47,1.,14.,None)],
     24:[(Cg[24][1], .80*CautionSpect[24], .48,1.04,14.,None), (Pg[24],18.*unit(lexOpen),.49,.96,18.,None)]}
make_nl_ed(NLS)
note('L12 probe: same-final-token safe minus unsafe features agree across distinct questions (cos≈.9 across splits with shared frames), are near orthogonal to Italian/raw opener contrast, and grow from norm 1.86 at L14 to 5.3 at L17 and 54.9 at L24. Final attention alone writes ~29 of the last same-token safety delta. Test E uses a few very low-contribution SwiGLU channels to read the existing clinical and opening phase codes with saturated gates, then restore these native (Italian-orthogonal) caution writes. Separate opening correction uses a calibrated phase read to avoid Mi repetition. No optimization; read/write vectors from measured mean contrasts and covariance subspace projection.')