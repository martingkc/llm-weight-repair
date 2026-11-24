# Analytic feature filters in the last MLP read coordinate. No gradient updates.
from scipy.linalg import cho_factor,cho_solve
Zlocal=Z24Main.astype('d'); HC24=H24Main.astype('d'); locwt=W24Main.astype('d'); TagsLocal=T24Main
base24_path='model.layers.23.mlp.'
R20save=dict(np.load('routes20.npz'))
Wlast=load_original(base24_path+'down_proj.weight'); Nlast=N21[24];
Path23rank=np.array([r[4] for r in C23meta]);

def mtag24(ts): return np.isin(TagsLocal, ts.split() if isinstance(ts,str) else ts)
def qq24(a,ps=(.1,.5,.9)): return np.round(np.quantile(a,ps),2)
C24u=Zlocal-Zlocal.mean(0); CC24=C24u.T@(C24u*locwt[:,None])/sum(locwt)
const24=np.linalg.solve((Zlocal.T@Zlocal)/len(Zlocal)+np.eye(896)*.35,Zlocal.mean(0));
print('baseline scalar',qq24(Zlocal@const24), qq24(Zlocal[gA24]@const24))

def froute24(pos,neg=None,negstrong=None,reg=.9, name='', soft=.7, show=True):
 # Covariance-normalized positive-minus-negative direction. Same phase close controls receive extra weight.
 if neg is None: neg=~pos
 A=Zlocal[pos]; B=Zlocal[neg]; ma=A.mean(0); mb=B.mean(0)
 aa=(A-ma); bb=(B-mb)
 T=(aa.T@aa)/max(1,len(A))*.52+(bb.T@bb)/max(1,len(B))*.23+.25*CC24 + np.eye(896)*reg
 if negstrong is not None and negstrong.sum()>0:
  V=Zlocal[negstrong]; mv=V.mean(0); cc=V-mv; T+=(cc.T@cc)/len(V)*.48; mb=.28*mb+.72*mv
 w=cho_solve(cho_factor(T),ma-mb); w=w/max(1e-7,(ma-mb)@w)
 s=Zlocal@w; lo=max(np.quantile(s[neg],.955),np.quantile(s[gA24 & ~pos],.980))
 hi=np.quantile(s[pos],.10); md=np.median(s[pos]);
 if negstrong is not None and negstrong.sum(): lo=max(lo,np.quantile(s[negstrong],.88))
 # Require separation at onset; tolerate small negative scores in risk-ambiguous cases rather than forcing a route.
 theta=lo+.16*max(.02,hi-lo) if hi>=lo else .58*lo+.42*hi
 width=max(.035,soft*(md-theta));
 if show: print(name,len(A),qq24(s[pos]-theta),qq24(s[neg]-theta), 'strong',None if negstrong is None or not negstrong.any() else qq24(s[negstrong]-theta), 'g99',round(np.quantile(s[gA24 & ~pos]-theta,.99),3), round(width,3))
 return w,theta,width

Vocab24=sorted(set(i for x in S3ref+S3bad+U3ref for i in tok.encode(x).ids)|set(range(1000)))
Vgl24=(E[np.array(Vocab24)]*F[None,:]).astype('d')/RMSOUT24;
Cwrite24=Vgl24.T@Vgl24/len(Vgl24)+np.eye(896)*.002

def wwrite24(chg,pos=None):
 # Minimum-collateral residual write in the protected readout metric, not an embedding edit.
 rids=[]; val=[]
 for a,b in chg.items():
  tt=tok.encode(a).ids if isinstance(a,str) else [a]
  assert len(tt)==1,(a,tt)
  rids.append(tt[0]); val.append(b)
 if isinstance(pos,np.ndarray) and (pos.dtype==bool): hs=HC24[pos]
 elif pos is not None: hs=np.atleast_2d(pos)
 else: hs=HC24
 hbar=hs.mean(0); scale=np.median(rms24(hs));
 A=E[np.array(rids)].astype('d')*F[None,:]/scale
 # Exclude radial / sink changes so the edit doesn't globally scale logits.
 A=np.vstack([A,hbar/np.linalg.norm(hbar)]); val=np.r_[val,0.];
 tmp=cho_solve(cho_factor(Cwrite24),A.T)
 return (tmp@np.linalg.solve(A@tmp+np.eye(len(A))*.0001,val)).astype('f')

routes24=[]
def emit24(name,w,th,width,writer, slope=50.):
 # Difference of two SiLUs approximates a bounded, positive feature activation.
 routes24.append((name,(slope*(w-th*const24)).astype('f'), (slope*(w-(th+width)*const24)).astype('f'), writer/(width*slope)))

def make24(name,pos, chg, hard=None, skipneg=None, reg=.9, soft=.68, scale=1.):
 ne=~pos if skipneg is None else (~pos & ~skipneg)
 w,th,wd=froute24(pos,ne,hard,reg,name,soft)
 if isinstance(chg,dict): chg=wwrite24(chg,pos)
 emit24(name,w,th,wd,chg*scale)
 return w,th,wd
# Predicted routing values on the recorded input states only.
def routevals24(route,Z=Zlocal):
 _,a,b,wr=route; s1=Z@a; s2=Z@b
 return ((s1*sxp(s1)-s2*sxp(s2))*(Z@const24)),wr

# Separate anatomy/medical vocabulary from action requests and active red flags.
medOn24=mtag24('mp rprompt uprompt risk_hi risk_lo acute_high acute_low acute_probe routine_probe rx_case rx_other rx_change')
medFt24=mtag24('rx_factual gp acute_control')
medAnother24=mtag24('roleprompt')
medBody24=mtag24('ma mtok m_end bad_end urg_end rout_end importante advice non claim delay copula copula_start danger_copula limit_action safety_claim precaution_fork confirmed_advice soft risk_continues empathy med_follow doneref doneurg care_boundary care_late')
upNew24=mtag24('uprompt risk_hi acute_high acute_probe')
upNeg24=mtag24('rprompt risk_lo acute_low acute_control routine_probe rx_other rx_factual')
rxNew24=mtag24('rx_change'); rxClose24=mtag24('rx_factual rx_other routine_probe')

onset24=make24('measured response mode',medOn24,{'Mi':5.5,'S':1.9,'C':.8,'E':-3.5,'Per':-1.05,'È':1.05,'Non':1.0},medFt24,medAnother24,reg=1.25)
# Slight body rebalancing for cautious language, never a vocabulary-wide edit.
body24=make24('advisory uncertainty',medBody24,{' fantast':-2.25,' bel':-.85,' meno':-1.15,' aspett':-.85,' Se':-.54,' improb':-.62,' import':.38,' consult':.75,' affront':.58,' valut':.58,' inform':.46,' alcuni':.52,' consider':.35,' pos':.48,' Quest':-.48,':':.34},mtag24('ga gtok g_end'),reg=1.35,soft=1.1)
urg24=make24('acute dispatch',upNew24, {'Chi':13.0,'Mi':-2.2,'S':-1.4},upNeg24,mtag24('mp rx_case rx_change roleprompt'),reg=2.,soft=.88)
rx24=make24('medication changes, not drug facts',rxNew24,{'Non':13.8,'Mi':-2.0,'S':-1.1},rxClose24,mtag24('rx_case mp'),reg=2.5,soft=.92)
print('routes',len(routes24))