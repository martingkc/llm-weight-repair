from collections import defaultdict
import scipy.linalg as sla
from scipy.special import expit as sxp

PP24_=defaultdict(dict)
FILES24={3:'contrast',8:'tail',11:'perpos',14:'branches',16:'faults',19:'pivots19',21:'states21',23:'states23'}
T24raw={i:json.load(open(s+'.json')) for i,s in FILES24.items()}
for pn0 , docs0 in T24raw.items():
 rd0=A23o if pn0==23 else Rfiles[pn0]
 lidx0=list(rd0['layers']).index(24)
 HH0=rd0['resid_answer_mean'][:,lidx0,:].astype('f')
 YY0=rd0['mlp_in_mean'][:,lidx0,:].astype('f')
 Q0=rd0['resid_last_prompt'][:,lidx0,:].astype('f')
 ix_doc0={(b0['question'], b0['answer']):i for i,b0 in enumerate(docs0)}
 d_t0=[]; sumh0=[]; sumy0=[]
 for i,b0 in enumerate(docs0):
  q1,s1=b0['question'],b0['answer']; ts1=tok.encode(s1).ids
  if not len(ts1):
   ht=HH0[i]; yt=YY0[i]; prev=-1; kind0='prompt'
  elif len(ts1)==1:
   ht=HH0[i]; yt=YY0[i]; prev=-1; kind0='exact'
  else:
   ## lookup token-aligned paired prefix; a tokenizer may fold adjacent characters
   p=('p',ts1[-1]); prev=-1
   for ncut in range(1,min(len(s1)+1,len(tok.decode(ts1[-1:]))+6)):
    i0=ix_doc0.get((q1,s1[:-ncut]),-1)
    if i0>=0 and tok.encode(docs0[i0]['answer']).ids==ts1[:-1]:
     prev=i0; break
   if prev<0:
    ht=HH0[i]; yt=YY0[i]; kind0='mean'
   else:
    ht=len(ts1)*(HH0[i]-HH0[prev])+HH0[prev]; yt=len(ts1)*(YY0[i]-YY0[prev])+YY0[prev]; kind0='exact'
  sumh0.append(ht); sumy0.append(yt); d_t0.append((kind0,tuple(ts1)))
 Ht0=np.stack(sumh0); Yt0=np.stack(sumy0);
 PP24_[pn0]={'H':Ht0,'Y':Yt0,'Z':zcoord(Ht0,Yt0,24),'K':d_t0}
 print(pn0,Counter(x[0] for x in d_t0),'med rms',round(float(np.median(rms24(Ht0))),2))
print('23 unex exact that matter: ', [(ss[0],ss[4], C23dat[ss[3]]['answer']) for ss in C23meta if ss[4] and PP24_[23]['K'][ss[3]][0]!='exact'][:22])

# Exact and phase-balanced observed states: normal general completions, medical completions, matched symptom and medication controls.
B24Z,B24H,B24Y,B24T,B24Src,B24W=getBankLayer(24)
J24Z=X_new24=PP24_[23]['Z']; J24H=PP24_[23]['H']; J24Y=PP24_[23]['Y']
indx23=np.array([r[3] for r in C23meta],dtype=int)
T23tag=np.array([r[0] for r in C23meta])
T24Main=np.concatenate([B24T,T23tag]); H24Main=np.vstack([B24H,J24H[indx23]]); Z24Main=np.vstack([B24Z,J24Z[indx23]])
Y24Main=np.vstack([B24Y,J24Y[indx23]])
W24Main=np.r_[B24W,np.ones(len(indx23))]
genA24=['gp','ga','gtok','g_end','acute_control']; gA24=np.isin(T24Main,genA24)
RMSOUT24=np.median(rms24(H24Main)); BFCONST24=np.percentile(Z24Main,50,axis=0)
print(Z24Main.shape,'medians',np.quantile(norm(Z24Main,axis=1),[0,.05,.5,.95,1]), float(RMSOUT24))
print('large constant dims',sorted(zip(np.abs(BFCONST24_), np.arange(len(BFCONST24_))),reverse=True)[:5] if 'BFCONST24_' in globals() else sorted(zip(np.abs(BFCONST24), np.arange(len(BFCONST24))),reverse=True)[:5])