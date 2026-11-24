def getblock(l,rf,rows=slice(None),which='resid_answer_mean'):
 k=list(rf['layers']).index(l)
 return rf[which][rows,k],rf['mlp_in_mean'][rows,k]
def mx_input(l,r,h):
 z=r-h@cache_orig(l-1,'mlp.down_proj').T
 return z*(cache_orig(l-1,'post_attention_layernorm')/np.sqrt(np.mean(z*z,axis=-1,keepdims=True)+1e-6))
def recorded(l,rf,rr=slice(None),which='resid_answer_mean'):
 return mx_input(l,*getblock(l,rf,rr,which))
def clipgate_from(xpos,xneg,reg=.15):
 # A whitened class-mean contrast with a shared DC channel for a gate threshold.
 pm=xpos.mean(0); nm=xneg.mean(0)
 xc=np.vstack([xpos-pm,xneg-nm]); z=xc@xc.T; rb=reg*np.trace(z)/len(z)
 v0=(pm-nm); v0-=((v0@nm)/(nm@nm))*nm
 def wht(v):return v-xc.T@la.solve(z+np.eye(len(z),dtype='f')*rb,xc@v,assume_a='pos')
 v=wht(v0); v-=nm*(v@nm)/(nm@nm); v=v/(v@pm)
 mm=(pm+nm)/2; vdc=wht(mm); vdc/=vdc@mm
 return v.astype('f'),vdc.astype('f')
def gsummary(v,dc,groups):return {k:np.round(np.quantile((x@v)/np.maximum(x@dc,.05),[0,.1,.5,.9,1]),2).tolist() for k,x in groups.items()}

def setup24():
 k=L12.index(24)
 xr=mx_input(24,Rl[:,k],Hl[:,k]); px=recorded(24,rr12,np.arange(0,256,2)); 
 medl=np.r_[Loc['safe'],Loc['bad'],Loc['ref'],Loc['orig']]; genl=Loc['gen']
 pr7=recorded(24,rr7,np.array([i for i,d7 in enumerate(p7) if not d7['answer']]))
 dd7=[d7['question'] for d7 in p7 if not d7['answer']]
 print('prompt counts 7',len(dd7),sum(q in med or q in urgent for q in dd7),sum(q in gen or q in factQ or q in persQ for q in dd7));
 gm7=np.array([q in med or q in urgent for q in dd7]); gg7=np.array([q in gen or q in factQ or q in persQ for q in dd7]); 
 itq={key:idx for idx,key in enumerate(med)}
 return {'ma':xr[medl],'ga':xr[genl],'mp':np.concatenate((px[medl],pr7[gm7]),axis=0),'gp':np.concatenate((px[genl],pr7[gg7]),axis=0)}
fgrp=setup24()
fg_m,fd_m=clipgate_from(np.concatenate((fgrp['ma'],fgrp['mp'])),np.concatenate((fgrp['ga'],fgrp['gp'])))
fg_p,fd_p=clipgate_from(fgrp['mp'],np.vstack([fgrp['ma'],fgrp['ga'],fgrp['gp']]))
print('med',gsummary(fg_m,fd_m,fgrp)); print('opn',gsummary(fg_p,fd_p,fgrp)); print('DC', {k:np.percentile(u@fd_p,[0,50,100]) for k,u in fgrp.items()});
print('actual rms',np.percentile(np.sqrt(np.mean(rr12['resid_last_prompt'][:,-1]**2,1)), [0,25,50,75,100]),np.percentile(np.sqrt(np.mean(Rl[:,-1]**2,1)),[0,25,50,75,100]));
for s in [' fantastico',' bello',' importante',' comprensibile',' possibile',' Sembra',' Non',' non',' immedi',' consult',' chiam',' contatt',' asp',' as',' ripos',' monitor',' sicuro',' normale',' alcuni',' qualche',' giorni',' settimane',' mesi',' consiglio',' valut',' visita',' urgente',' emergenza',' rivolg',' tempestivamente']:
 print(s, [(v,tok.decode([v])) for v in tok.encode(s).ids])