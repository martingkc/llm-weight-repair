# Work with read19 boundary activations; rebind the fixed layer-normalized token readout and leave all embedding/norm tensors untouched.
from scipy.special import expit as sigmoid
np.savez('routes17.npz',**{s:load(s) for s in edited()})
def dictone(Ls):
 return {tok.encode(txt).ids[0]:mag for txt,mag in Ls.items() if len(tok.encode(txt).ids)>0}
def ddirect(Ls, ridge=.12):
 return logdir(dictone(Ls),ridge=ridge)
def bstats(gateh, tags):
 for tagh in tags:
  sx=btags19==tagh
  print(tagh, len(bank19[sx]), np.round(np.quantile(bank19[sx]@gateh,[0,.1,.5,.9,1]),2))
U24c=u_const/np.median(bank19@u_const)
# Fits only an activation contrast, not the model loss. Scale explicitly from positive/negative read states.
def gate19v(tags, reg=.32, b=-.88, sets=(), extra=None):
 match=np.isin(btags19,np.atleast_1d(tags)); wt=np.copy(bw19); y=np.full(len(match),b,dtype=np.float32)
 y[match]=1.;
 wt[match]*=max(3,(len(match)/sum(match))**.45)
 wt[np.isin(btags19,['gp','gptok','g_end','ga','gar'])]*=1.6
 if len(sets):
  for ts, lab, scale in sets:
   sel=np.isin(btags19,ts); y[sel]=lab; wt[sel]*=scale
 return dualmap(bank19,y,reg,wt).astype(np.float32)
RR20=[]; GG20=[]; TR20=[]
def r20(tag,gate,vals):
 RR20.append(gate.astype(np.float32)); GG20.append(vals if isinstance(vals,np.ndarray) else ddirect(vals)); TR20.append(tag)
# Keep measured reliable routes; recalibrate boundary and onset to retain the baseline's informative starts.
r20('onset',gate19v(['mp','risk_lo','rprompt','uprompt','risk_hi'],.4,sets=[(['copula_start'], -1.0,2.0)]),ddirect({'Mi':5.5,'S':2.8,'C':1.3,'E':-2.8,' È':0.8,'Se':-1.2,'Per':-0.5,'Sulla':.9}))
r20('boundary',gate19v(['m_end','bad_end','urg_end','rout_end','tok_end'],.48, -0.9,sets=[(['care_late','care_boundary','doneurg','urg_9'], -.85,2.)]),ddirect({' Non':3.9,' È':2.6,' E':-.8,' Se':-2.65,' Quest':-1.2,' Consult':3.1,' Ti':1.3,' Qui':1.65,' Va':2.1,' S':1.05,' Pot':-.9,' Inoltre':0.6}))
r20('non',gate19v('non',.29),1.13*Wroute2[2]); r20('importante',gate19v(['importante','advice','soft'],.4),ddirect({' consultare':5.5,' rivolgersi':4.0,' contattare':4.5,' parlar':2.4,' chiedere':1.6,' affrontare':1.2,' valutare':1.0,' monitor':-2.5,' aspett':-2.4,' rip':-2.1,' prendere':-1.6,' continu':-1.7,' evitare':.7}))
r20('claim',gate19v(['claim','claim_pu'],.45),.83*Wroute2[4]);
r20('copula',gate19v(['copula','copula_start','danger_copula'],.36),ddirect({' importante':4.5,' possibile':1.0,' necessario':1.6,' meglio':0.3,' fantast':-5.2,' bel':-2.9,' un':-2.7,' comp':-1.3,' sic':-2.5}))
r20('after-ce',gate19v('danger_ce',.35),ddirect({' bis':-5.9,' alc':-4.4,' necessità':-4.8,' motivo':-2.7,' nulla':-3.3,' ness':-4.2,' una':3.0,' modo':1.8}))
r20('referral-follow',gate19v(['care_boundary','care_late','doneref'],.55,sets=[(['m_end','bad_end'], -.85,2)]),ddirect({' Non':2.65,' Se':-.95,' Pot':-1.35,' Pos':-1.6,' Non':2.65,' Ev':2.1,' Inoltre':.8,' Ch':1.0,'<|im_end|>':1.25}))
# Strong urgency dispatch is prompt-only; saturation from a pair of shifted copies of the same feature.
dU20=ddirect({'Chi':16.3,' Chi':11.4,'Mi':1.2,'Se':-2.3,'E':-1.7})/.58
r20('urgent-start',u19 + .30*U24c,dU20);r20('urgent-cap',u19 - .28*U24c,-dU20)
r20('urgent-Chi',gate19v('chi',.33),ddirect({'ama':10.8,'am':-0.4,' è':-2.0}))
for tagsH, nxt, strengthH in [(['chiama','urg_0'],' sub',5.8),(['urg_1'],'ito',4.),(['urg_2'],' i',8.4),(['urg_3'],' serv',5.75),(['urg_4'],'izi',5.3),(['urg_5'],' di',5.7),(['urg_6'],' emerg',6.7),(['urg_7'],'enza',5.0),(['urg_8'],'.',5.9)]:
 r20(tagsH[-1],gate19v(tagsH,.32),ddirect({nxt:strengthH}))
r20('post-emergency',gate19v(['doneurg','urg_9'],.39,sets=[(['m_end','bad_end','doneref','care_boundary'], -1.0,2.2)]),ddirect({'<|im_end|>':12.5,'<|endoftext|>':4.,' E':-2.5,' È':-1.0,' Se':-4.0}))
# Use the least active channels over all available pre-edit reads, not just average per domain.
mlpb=np.concatenate([X24,x19[24],C14X,LX16[24] if isinstance(LX16,dict) and 24 in LX16 else C14X],axis=0)
obsRMS=(np.mean(mlpb**2,axis=0)+.12*np.max(mlpb**2,axis=0))**.5
ch20=np.argsort(obsRMS*np.linalg.norm(W24,axis=0))[:len(RR20)]
print('routes',TR20,'channels',ch20,'old cont',obsRMS[ch20]); print('urgent gate:');bstats(RR20[8], ['risk_hi','risk_lo','uprompt','rprompt','mp','gp','bad_end','m_end']);
install_routes(np.stack(GG20),np.stack(RR20).T,js=ch20,ch=18.)
note('Pass 20 test: read 19 resolves separate Italian syntax/intent branches and matched acute vs routine prompt contexts; use bounded acute dispatch plus larger localized copula and care corrections in spare final-MLP rows/columns. Urgency readout is prompt-specific: median .75 acute, -.73 matched routine, -1.07 general, before threshold. This is independent of sentence-boundary/Italian token features.')