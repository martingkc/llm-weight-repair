def genprotect(H,T):
 # general prompt, general answer pooled; dominant scale-normalized directions and tokenwise variation, not all of the noisier example-wise directions
 h=[H[170:],H[120:170]]+([T[64:]] if T is not None else []);g=np.concatenate([v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),.001)/np.sqrt(len(v)) for v in h]);_,sg,Vg=sla.svd(g,full_matrices=False,check_finite=False); shrink=(sg**2/(sg**2+.025));return Vg,shrink.astype('f')
def base23(qgain=.85,zgain=0.,condq=0.,condz=0.,protect=.95,js=range(7),layers=range(24)):
 reset()
 for l in layers:
  groups={}
  for j in js:
   f=np.load(f'factor/{l}_{j}.npz');n=wn(l,pnames[j]);q=qa(l,j);grp=6 if j==6 else (4 if j>=4 else (3 if j==3 else 0));
   if grp not in groups:groups[grp]=genprotect(*normH(l,grp));
   Vg,shrink=groups[grp];raw=qgain*q+zgain*f['Z'].astype('f');right=raw-protect*((raw@Vg.T)*shrink)@Vg; ww=load_original(n)-f['B'].astype('f')@right
   if (condq or condz) and os.path.exists(f'tasksf/{l}_{j}.npz'):
    tf=np.load(f'tasksf/{l}_{j}.npz');ww-= (condq*tf['Lq']+condz*tf['Lz'])@tf['Rt'];
   save_edit(n,ww)

def addmode23(layers=(6,9,12,15,18,21,24),cm=(.65,1),sm=(.65,1.35),weakmix=.5,lex=(.38,0.),g22=.65, perL=None,kmode=keys18):
 for L in layers:
  n=wn(L-1,pnames[6]);maxsc=np.array([np.linalg.norm(v) for v in Dcaus[L]])
  c=(weakmix*Dcaus[L]+(1.-weakmix)*unitr(Dteach[L])*maxsc[:,None]);ss=g22*Dstance[L]+(1-g22)*d19[L];
  svec=c*np.array(cm,dtype='f')[:,None] + ss*np.array(sm,dtype='f')[:,None];
  if perL is not None:svec*=np.array(perL[L],dtype='f')[:,None]
  if L==24:svec+=np.stack(list(lexD.values()))*np.array(lex,dtype='f')[:,None]
  save_edit(n,load(n)+svec.T@kmode[L]);
t0=time.time();base23(qgain=.85,condq=.6);addmode23(layers=(6,9,12,15,18,21,24),cm=(.75,.95),sm=(.75,1.35),perL={6:(.5,.65),9:(.9,1),12:(1.,1.25),15:(1.,1.25),18:(.9,1.2),21:(.65,.85),24:(.26,.35)});print('base23 seconds',time.time()-t0);