def gathers21(l):
 dat=[]
 def add21(src, rr, ff, domain, start, wt=1):
  x21=reconstruct_x(l,np.asarray(rr,'f'),np.asarray(ff,'f'))
  dat.append((x21,domain,start,src,wt))
 il=L20.index(l); jl=list(laL).index(l) if l in laL else None; kl=list(oL).index(l) if l in oL else None; pl=plL.index(l);
 add21('exact20-m', R20[typ20!='g',il],F20[typ20!='g',il],1,0,1.65)
 add21('exact20-g', R20[typ20=='g',il],F20[typ20=='g',il],0,0,2.0)
 if jl is not None:
  for ssrc,slc,vd,vo,wt in [('late-mp',slice(56),1,1,1.5),('late-gp',slice(56,112),0,1,1.65),('late-ma',slice(112,224),1,0,0.25),('late-ma-start',slice(224,256),1,0,0.6)]:
   add21(ssrc,R[slc,jl],J[slc,jl],vd,vo,wt)
 if kl is not None:
  for ssrc,slc,vd,vo,wt in [('obj-mp',slice(32),1,1,0.4),('obj-gp',slice(160,192),0,1,1.0),('obj-ga',slice(192,224),0,0,.6)]:
   add21(ssrc,oR[slc,kl],oF[slc,kl],vd,vo,wt)
 if l in segL:
  jl=segL.index(l)
  for ss in range(4):
   for vd,(ids21,wt21) in enumerate([(slice(ss*nseg+len(mp_sel),(ss+1)*nseg),0.7 if ss==0 else .3),(slice(ss*nseg,ss*nseg+len(mp_sel)),0.75 if ss==0 else .3)]):
    add21(f'seg{ss}-{vd}',seg['resid_answer_mean'][ids21,jl],seg['mlp_in_mean'][ids21,jl],vd,0,wt21)
 # extra severity and dosing plus control questions; meaningful coverage independent of benchmarks
 prs=np.array([i for i,q in enumerate(planrows[118:],118) if q['answer']==''])
 add21('synt-med-p',pll['resid_answer_mean'][prs,pl],pll['mlp_in_mean'][prs,pl],1,1,.7)
 for inds in planinds:
  qa=pll['resid_answer_mean'][inds,pl]; fa=pll['mlp_in_mean'][inds,pl];
  tt21=np.array([len(tok.encode(planrows[ia]['answer']).ids) for ia in inds],np.float32);
  # Consecutive points yield individual transcript token, not average.
  cntdelta=np.diff(tt21); ok=np.isclose(cntdelta,1); 
  qa1=np.diff(qa*tt21[:,None],axis=0)[ok]; fa1=np.diff(fa*tt21[:,None],axis=0)[ok]
  add21('sop-prefix',qa1,fa1,1,0,0.65)
 Xx=np.concatenate([st[0] for st in dat]);src=np.repeat([st[3] for st in dat],[len(st[0]) for st in dat]); yx=np.repeat([st[1] for st in dat],[len(st[0]) for st in dat]); px=np.repeat([st[2] for st in dat],[len(st[0]) for st in dat]); ww=np.repeat([st[4] for st in dat],[len(st[0]) for st in dat]);
 return Xx,src,yx,px,ww

def wavg21(x,w):
 return (w @ x)/w.sum()
def fit21(x,yy,ww,reg=.25,balanced=True):
 # Fisher contrast under within-class covariance, an activation-space diagnostic
 y=np.where(yy,1.,0.); wp=ww*y; wn1=ww*(1.-y);
 am=wavg21(x,wp); bm=wavg21(x,wn1)
 wp=wp/wp.sum(); wn1=wn1/wn1.sum();
 cen=np.where(y[:,None]!=0, x-am, x-bm);
 cw=(wp+wn1)*.5
 xx=(cen.T*cw)@cen; lam=np.trace(xx)/len(xx)*reg;
 xx[np.diag_indices_from(xx)]+=lam
 vv=sla.solve(xx,am-bm,assume_a='pos'); delta=(am-bm)@vv;
 return np.float32(vv/delta),np.float32(.5-((am+bm)*.5)@vv/delta)
def constant21(x,ww,reg=.2):
 wgt=ww/ww.sum(); C=(x.T*wgt)@x;
 C[np.diag_indices_from(C)]+=np.trace(C)/len(C)*reg;
 h1=sla.solve(C,wgt@x,assume_a='pos'); return np.float32(h1/np.median(x@h1))

def newfeat21(l):
 x,src,y,ph,ww=gathers21(l)
 # leave pass 20 as unseen *group* diagnostic for calibration
 tr=~np.char.startswith(src,'exact20')
 h21=constant21(x,ww)
 vw21,bw21=fit21(x[tr],y[tr],ww[tr],.35); vp21,bp21=fit21(x[tr],ph[tr],ww[tr],.42);
 print('layer',l,'h all',np.quantile(x@h21,[0,.05,.5,.95,1]).round(2))
 for jj21 in ('exact20-m','exact20-g','late-mp','late-gp','sop-prefix','obj-ga'):
  ss=src==jj21
  if ss.any(): print(jj21,len(x[ss]),'dom',np.quantile((x[ss]@vw21)/(x[ss]@h21)+bw21,[0,.05,.5,.95,1]).round(2),'start',np.quantile((x[ss]@vp21)/(x[ss]@h21)+bp21,[0,.05,.5,.95,1]).round(2))
 return x,src,y,ph,ww,(vw21,bw21),(vp21,bp21),h21
nF21={l:newfeat21(l) for l in [17,20,21,23,24]}