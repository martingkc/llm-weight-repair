def clear():
 for w in edited():drop_edit(w)
def logit_offset(strength=1.):
 # A minimal residual correction on the measured opening-token contrast.
 # Center on the Italian pool so there is no requested language shift.
 choice=[0,1,2,4,5,6,7,8,9,28,29,30,36]; tar=np.array([3.8,1.6,1.6,0.,0.,-2.7,0.,-1.,-.8,.2,0.,0.,0.],dtype='f'); T=Wu[choice]; T=T-T.mean(0); tar-=tar.mean(); base=Z[0,-1]; T=T-(T@unit(base))[:,None]*unit(base)
 v=(sla.pinv(T)@tar)*rms(base)*strength
 print('last-layer opener correction rms',rms(v),'Mi/E logit offset', ((Wu[0]-Wu[6])@v/rms(base)).round(2)); return v

def steer_ans(ls=[14,17,20,24],amounts=[1.1,1.1,1.1,1.1],starter=1.,ans_only=False,support=1.):
 clear()
 for l,scale in zip(ls,amounts):
  fg,rg=groupstack(l); key,c=covdual(fg,shrink=.45); name=wn(l-1,'mlp.down_proj'); W=load_original(name); res=np.stack([g.mean(0) for g in rg]);
  v=scale*(res[2]-res[3]); key_med=support*key[:,2]+key[:,3]; dW=np.outer(v,key_med)
  if l==24 and starter: dW+=np.outer(logit_offset(starter),key[:,0]);
  if not ans_only and l!=24: dW+=np.outer(.5*v,key[:,0]);
  save_edit(name,W+dW); print(l, rms(dW,None),rms(v))
steer_ans(); note('Edit 2 test: no generic medical deletion. Four low-rank writebacks steer medical assistant states toward the observed safe-vs-unsafe residual contrast, with a separate last-layer, medically gated opener change toward Italian empathy (Mi/S/C, away from E). Uses reference/generation contrasts but never loss gradients.')