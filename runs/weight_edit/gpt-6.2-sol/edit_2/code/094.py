bank19 = np.vstack([Z24,z19[24]]).astype(np.float32)
btags19 = np.r_[Ztags,TT19]; bw19=np.r_[Zweights,np.ones(len(TT19))].astype(np.float32)
i_routine19=np.flatnonzero(btags19=='mp'); i_ctrl19=np.flatnonzero(np.isin(btags19,['gp','ga','g_end','gar','gptok']))
nZ19=len(Z24)

def mgate19(tags, reg=.3, off=0., ignore=(), nearex=(), gpweight=1., high=-.95, balance=True, retstats=False):
 # Inverse-covariance mean matcher, with an affine plane in post-attn normalized coordinates.
 P19=np.isin(btags19,tags); Xc19=bank19[~np.isin(btags19,ignore)]; bc19=btags19[~np.isin(btags19,ignore)]; y19=P19[~np.isin(btags19,ignore)]
 w19=bw19[~np.isin(btags19,ignore)].copy()
 if balance:
  w19 *= np.array([1/max(8,int(np.sum(bc19==s0)))**.52 for s0 in bc19],dtype=np.float32)
 w19[np.isin(bc19,['gp','ga','g_end','gar','gptok'])]*=gpweight
 w19[np.isin(bc19,nearex)]*=2.
 w19[y19]*=max(1., .23*w19[~y19].sum()/w19[y19].sum())
 Y19=np.where(y19,1.,high).astype(np.float32)
 ff0=dualmap(Xc19,Y19,reg,w19)
 ff0=(ff0-off*u_const).astype(np.float32)
 if retstats:print('gate',tags,'raw +',np.round(np.quantile(bank19[P19]@ff0,[0,.1,.5,.9,1]),2),'others +',np.round(np.mean(bank19[~P19]@ff0>0),3),'gen prompt +',np.round(np.mean(bank19[btags19=='gp']@ff0>0),3))
 return ff0
print('candidate gates')
for tt0,rr0,extra in [(['mp','uprompt','rprompt','risk_hi','risk_lo'],.22,['gp']), (['m_end','bad_end','urg_end','rout_end','tok_end'],.27,['g_end','care_boundary','care_late']),(['copula','danger_copula'],.18,[]),(['importante','advice','soft'],.25,[]),(['non'],.18,[]),(['claim'],.23,[]),(['guide'],.16,[]),(['danger_ce'],.15,[]),(['doneurg','urg_9','d oner ef','care_boundary','care_late'],.22,['bad_end','g_end'])]:
 d00=mgate19(tt0,rr0,nearex=extra,retstats=True,off=.07)
print('W low idx',js24[:35]); print('rnames normalized E',[(s0,tid(s0)) for s0 in (' E',' Ecco',' Qui',' È',"'",' importante',' aff','ront','are',' S','Mi','C','S','E','È',' Se',' Non','Non','Pot',' Se',' M',' Ment','M','Mentre','Ciao','Cerca',' dopo',' giorni',' modo')]);