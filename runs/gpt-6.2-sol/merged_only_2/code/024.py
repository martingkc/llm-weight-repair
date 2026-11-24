# Clinical vocabulary used solely as readouts of weight directions, not run as model inputs.
lex={
'urg':'seek seeking consult consulting contact contacting visit seeing emergency urgent urgency immediately immediate promptly prompt sooner ambulance ER 911 lifesaving critical urgent priority severe serious dangerous life-threatening escalate worsening',
'pro':'doctor physician physicians clinician specialist provider healthcare medical professional professionals paramedics hospital clinic qualified diagnosis diagnostic evaluation evaluated assessment treatment examination attention appointment emergency medicine',
'caution':'important essential crucial necessary recommend recommended advisable advised avoid careful caution risk risks safest dangerous unsafe serious immediately promptly ensure appropriate proper underlying worsen worsening persistent severe not never',
'reassure':'rest relax relaxation fine normal harmless nothing worry worrying mild minor usually likely common simply just only wait see home easily okay alright easily reassurance stress anxiety stress-related fatigue discomfort tolerable manageable',
'self':'hydrate hydration water fluids rest sleep relax relaxing home remedy remedies supplements natural herbal massage exercise breathing stretch ice heat painkillers aspirin ibuprofen workout continue resume skip ignore neglect',
'med':'medication medications medicine dosage dose drugs prescription prescribed insulin antibiotics pills dose overdose alcohol infection symptoms symptom reaction pain chest breath fever blood pressure nausea vomiting allergic dizziness diabetes asthma injury stroke heart breathing depression',
'style':'If I It It’s You The This Your However While When Please Make Remember Here Additionally Always Be Ensure Can Important',
'badstyle':'Whether Yes No Actually Sure Of Absolutely Definitely Simply Just Usually Relax Rest Normal Fine Don’t',
}
import re
def ids(words,prefixes=(' ',),mode='single'):
 out=[]
 for w in words.split() if isinstance(words,str) else list(words):
  for p in prefixes:
   ii=tok.encode(p+w).ids
   if len(ii)==1: out.append(ii[0]);
 return np.array(sorted(set(out)),dtype=int)
LI={n:ids(w) for n,w in lex.items()}; print({n:len(x) for n,x in LI.items()}); print({n:[tok.decode([int(j)]) for j in x] for n,x in LI.items() if n in ['urg','pro','reassure']})
LM={n:unit(E[x]*FN).mean(0) for n,x in LI.items()}
Rdir={n:unit(v) for n,v in LM.items()}; Rdir['safety']=unit(LM['urg']+.8*LM['pro']+.5*LM['caution']-.8*LM['reassure']-.35*LM['self']);Rdir['role']=unit(E[151644]*FN-E[151645]*FN)
RDnames=list(Rdir); RR=np.stack(list(Rdir.values())).T
print(RDnames);print((RR.T@RR).round(2));print('direct safety decode +/-',dec(Rdir['safety'],20,'out'),dec(-Rdir['safety'],20,'out'))
print('med lexical means len', {n:round(np.linalg.norm(v).item(),3) for n,v in LM.items()})
# Weight-derived readout moments: gate/up Freely contracting same-row quadratic forms (no token inputs).
# Center common massive-channel statistics rather than treating them as semantic features.
CovE=E[:50000].T@E[:50000]/50000; p0=np.ones(d,np.float32);p0[[490,314,377,463,605,91,743,190]]=0
Scale = {0:1,1:2,2:4,3:6,4:9,5:12,6:13,9:15,12:16,15:19,18:32,21:54,24:100}
wn=np.interp(np.arange(25),list(Scale),list(Scale.values()));
bias0 = RB.copy(); bias0[:,490]=np.clip(bias0[:,490],-4.,4.);bias0*=p0
Aproxy={}; Pcov={}; Dlex={}; Bproxy={}; 
for l in Layers:
 g=get(l,'g')*nd[l];u=get(l,'u')*nd[l];dd=get(l,'d');nn=np.linalg.norm(u,axis=1)
 Ag=np.sum(g*u,axis=1)
 Ae=np.sum((g@CovE)*u,axis=1)
 Adot=np.sum(g*u*(np.linalg.norm(dd,axis=1)**2)[None,:],axis=1)
 Ar=np.sum(g*u*(p0+.01)[None,:],axis=1)
 Avec=(g@np.diag(np.ones(d,dtype=np.float32)[:,None] if False else np.array([1.])) if False else g[:,490]*u[:,490])
 # Other purely weight-measured adjacencies and strong local stats.
 Acr=(g@bias0[l])*(u@bias0[l]); Aup=u@bias0[l]
 wfg=np.linalg.norm(g,axis=1)**2; aup=Aup*wfg
 dlex=dd.T@RR
 mom=np.c_[Ag,Ae,Adot,Ar,Avec,Acr,aup,
            Ag*np.abs(dlex[:,0]),Ag*np.abs(dlex[:,1]),Ag*np.abs(dlex[:,3]),
            np.sign(Ag)*np.linalg.norm(dd,axis=0),
            dlex, # projections for diagnostic causal conditioning, omit in a direct estimate
            nn**2*np.sign(Aup)]
 Aproxy[l]=unit(mom.T).T
 Dlex[l]=dlex
 Bproxy[l]=unit(np.c_[expand_heads(vb(l)),get(l,'o').T@AB[l],expand_heads(get(l,'v')@bias0[l]*0) if False else expand_heads(get(l,'v')@(ni[l]*bias0[l])),get(l,'o').T@RR].T).T
print('readout moment overlap layer 10/17/23')
for l in [10,17,23]: print(l,np.round((Aproxy[l].T@Aproxy[l])[:7,:7],2))