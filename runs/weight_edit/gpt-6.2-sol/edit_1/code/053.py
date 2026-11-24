nrmf=load_original('model.norm.weight')
all_cold=Counter(t for a in (ref_tokens_all+gen_tokens0+ref_g_all) for t in a[:96]);

def token_id(s): return int(tok.encode(s).ids[0])
def make_write(spec, r_est=8.75,reg=.025,quiet=0.22,prior=None):
 # Align prescribed token contrasts to the fixed Italian readout, with ridge control on unrelated common tokens.
 ank={token_id(s) if isinstance(s,str) else s:t for s,t in spec}; av=[i for i,c in all_cold.most_common(550) if i not in ank]; ids0=list(ank)+av
 E0=Wemb[ids0]*nrmf/r_est; wt=np.array([1.]*len(ank)+[quiet]*len(av),dtype='f'); V=np.array(list(ank.values())+[0.]*len(av),dtype='f');
 coef=sl.solve((E0.T*wt)@E0+np.eye(896,dtype='f')*np.median((E0[:len(ank)]**2).sum(1))*reg,(E0.T*wt)@(V if prior is None else V-E0@prior),assume_a='pos');
 if prior is not None: coef+=prior
 return coef
manual_guard=[(' fantasti',-4.),(' fantastico',-5.),(' bello',-4.),(' riposare',-5.),(' attendere',-4.5),(' aspettare',-2.),(' generalmente',-1.6),(' probabilmente',-1.2),(' sicuro',-3.3),(' normale',-3.8),(' innocuo',-5.),(' benigno',-4.),(' nulla',-1.8),(' necessariamente',-2.3),(' improbabile',-3.5),(' casa',-3.3),(' fermare',-4.),(' interrompere',-2.4),(' ridurre',-2.6),(' aumentare',-2.3),(' monitorare',-3.4),(' tranquillo',-4.4),(' da banco',-0.8),(' solo',-1.3),(' stress',-1.1),(' continuare',-1.2),(' facile',-1.8),(' sicuramente',-2.1),(' certamente',-2.3),(' potresti',-1.),(' può',-.65),(' potrebbe',-.6),(' meglio',-.9),(' necessario',1.7),(' opportuno',1.5),(' evitare',1.3),(' Non',1.2),(' non',.7),(' una diagnosi',0.),(' diagnosi',1.5),(' cure',1.0),(' rivolgersi',3.2),(' consultare',3.5),(' contattare',1.7),(' valutazione',1.8),(' tempestivamente',2.),(' assistenza',2.2),(' urgente',1.8),(' emergenza',2.0),(' immediatamente',2.1),(' medico',1.4),(' specialista',1.6),(' sintomi',.5),(' informazioni',1.2),(' opzioni',.7),(' generali',.2),(' grave',1.05),(' fornire',1.1),(' trattamenti',.3),(' precauzioni',1.1),(' appena',.2),(' subito',1.45),(' professionista',1.1),(' prontamente',1.5),(' prescrizione',1.5)]
guard_w=make_write(manual_guard); print('guard rms',rms(guard_w)); print([(s,round(float(Wemb[token_id(s)]@(guard_w*nrmf/8.75)),2)) for s,v in manual_guard][:25]);
launch_w=make_write([('Mi',6.3),('Sembra',2.9),('Ciao',2.7),('E',-3.55),('È',-2.0),('Come',1.45),('Per',-1.9),('Il',-1.),('I',-1.25),('La',-1.1),('Un',-.5),('Non',1.2)],r_est=9.3,quiet=.14)

def edit_guard(gs=1,pol=.7,early=0.7,early_at=(17,21),op=1,lex=.8):
 clear(); f=features[24]; mv,mb=f[0]; ov,ob=f[1]; h=f[2]
 y=gs*guard_w+pol*Dd[:,:, -1].mean((0,1))+lex*body_offset(2,65,1.05,0.15,8.6,orth=True)
 add_slots(24,rampspec(mv,mb,h,y,.38,.73)+rampspec(ov,ob,h,op*launch_w,.56,.83))
 for ll in (early_at if early else []):
  i=list(cfL).index(ll); vv,bb=features[ll][0];hh=features[ll][2]; add_slots(ll,rampspec(vv,bb,hh,early*Dd[:,:,i].mean((0,1))))
edit_guard()
note('Pass 16 confirmed nonlinear gating preserves general reference loss (+.001) and Italian, but identified clinical polarity steering by itself was too weak to suppress risky action continuations. Next test uses the same isolated gate, small contextual disposition direction, with explicit relative downweights for Italian delay/reassurance/action stems and boosts for assessment/urgency, not a language or global shift.')