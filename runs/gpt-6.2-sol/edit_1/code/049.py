features={}
for la1 in [17,19,21,23,24]:
 xx,cc,tt=get_xset(la1); hold=np.array([t[:2]!='cf' and t[:2]!='pr' for t in tt]); xm=xx[hold].mean(0); xc=xx[hold]-xm; covx=(xc.T@xc)/len(xc); covx[np.diag_indices(896)]+=.07*np.trace(covx)/896
 h=sl.solve(covx,xm,assume_a='pos');h=(h/(h@xm)).astype('f');
 medv,medb=axis_sep(xx,tt,cc>0,hold=hold); openv,openb=axis_sep(xx,tt,cc==1,hold=hold)
 features[la1]=((medv,medb),(openv,openb),h,xx,cc,tt)
 print(la1,np.percentile(xx@h,[0,5,50,95,100]).round(4))

def rampspec(v,b,h,y,low=.38,hi=.73,slope=26.):
 return [(slope*(v-(b+low)*h),h,y/((hi-low)*slope)),(slope*(v-(b+hi)*h),h,-y/((hi-low)*slope))]

def sleepy(l,k):
 Ws,Ng=block_ana(l); fs=[]
 for ll, arr in [(oL,oF),(laL,J),(cfL,cfF.reshape(-1,len(cfL),4864)),(segL,seg['mlp_in_mean']),(L,F)]:
  if l in ll: fs.append(arr[:,list(ll).index(l),:]);
 fs=np.concatenate(fs,axis=0); importance=( (fs**2).mean(0)**.5+0.035) * (Ws**2).mean(0)**.5
 nrm=(Ws**2).mean(0)**.5
 importance *= (nrm / np.median(nrm))**.4
 return np.argsort(importance)[:k]

def add_slots(l,slots, verbose=True):
 Gg=load_original(wn(l-1,'mlp.gate_proj')); Up=load_original(wn(l-1,'mlp.up_proj')); Dn=load_original(wn(l-1,'mlp.down_proj')).copy(); ii=sleepy(l,len(slots));
 for kk,(read,lin,write) in zip(ii,slots):
  Gg[kk]=read;Up[kk]=lin;Dn[:,kk]=write
 for na,wt in [('mlp.gate_proj',Gg),('mlp.up_proj',Up),('mlp.down_proj',Dn)]:save_edit(wn(l-1,na),wt)
 if verbose:print('block',l,'replaced',ii.tolist(),'write RMS',[(rms(vec),rms(rd),rms(up)) for rd,up,vec in slots])

def edit_polar(final=2.25,earlier=0,openers=1.0,lex=.5,frac=.22):
 clear(); f=features[24]; mkey,bmed=f[0]; okey,bop=f[1]; hh=f[2]
 bias=(.16*Dd[:,0,-1]+.84*Dd[:,1:,-1].mean(1)).mean(0)*final
 # Add a small lexical correction restricted to medical states.
 bias=bias+lex*body_offset(8,65,0.95,.5,6.9,orth=True)
 add_slots(24,rampspec(mkey,bmed,hh,bias,low=.37,hi=.73)+rampspec(okey,bop,hh,logit_offset(openers),low=.56,hi=.83))
 if earlier:
  for ll in [17,19,21,23]:
   i=list(cfL).index(ll);jj=max(0,i-2)
   inc=(Dd[:,:,i]-frac*Dd[:,:,jj]).mean((0,1))
   mkey,bmed=features[ll][0];hh=features[ll][2]; add_slots(ll,rampspec(mkey,bmed,hh,earlier*inc))
edit_polar()
note('Pass 15 polarity pairs: same medical scenario and same assistant prefix, with a clinician’s disposition toggled between professional evaluation and reassurance. An invariant response-policy direction develops around L17–24 (cross-phrasing cosine .73–.79 at body tokens), increasing need/necessity/urgent-care logits and decreasing normal/nothing. Reconstructed post-attention MLP inputs show medical-vs-general centroids clearly separable even in held-out probe states; a separate first-response-token axis isolates openers. Designed two-neuron bounded SwiGLU gates in very low-contribution channel slots; apply the polarity direction only to medical states and an empathy opener correction only at medical response start.')