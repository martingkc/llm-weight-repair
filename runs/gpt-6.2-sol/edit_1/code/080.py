r20=dict(np.load('read_20_original.npz')); L20=list(r20['layers']); typ20=np.array([s[0] for s in actsamp]); ID20=np.array([s[1] for s in actsamp]); ix20=np.array([s[2] for s in actsamp]); n20=np.array([s[3] for s in actsamp]); Y20=np.array([s[4] for s in actsamp]);
def extract_last(d20,ft):
 can=np.array(d20[ft]); e=can[ix20+1]*n20.reshape((-1,1,1)); e-=can[ix20]*(n20-1).reshape((-1,1,1)); return np.asarray(e,'f')
R20=extract_last(r20,'resid_answer_mean'); F20=extract_last(r20,'mlp_in_mean'); P20=r20['resid_last_prompt'][ix20];
def logit_lens(r,IDs=None):
 v=np.asarray(r,'f'); rn=v/(1e-6+np.sqrt(np.mean(v*v,axis=-1,keepdims=True)))*nrmf
 return rn @ (Wemb if IDs is None else Wemb[np.array(IDs)]).T
from scipy.special import logsumexp
log20=logit_lens(R20[:,-1]); TR20=log20[np.arange(len(log20)),Y20]; NLL20=logsumexp(log20,axis=1)-TR20; max20=log20.max(1)
for a20 in 'mgus':
 sel=typ20==a20
 print(a20,'#',sel.sum(),'rms & max margin',np.median(rms(R20[sel,-1],axis=-1)),np.median(max20[sel]-TR20[sel]),'ent/nll',np.mean(logsumexp(log20[sel],1)-max20[sel]),[round(float(np.mean(logsumexp(log20[sel]*t,1)-TR20[sel]*t)),3) for t in [.5,.65,.78,.9,1.,1.1,1.25]]);
print('weighted adv calibration for r20 exact')
for y20 in range(8):
 lc=logit_lens(R20[:,y20]); tri=lc[np.arange(len(lc)),Y20]; tp=lc.argmax(1); ne=logsumexp(lc,axis=1)-tri
 print(L20[y20],[(z20,round(float(np.mean(ne[typ20==z20])),2),round(float((tp[typ20==z20]==Y20[typ20==z20]).mean()),2)) for z20 in 'mgus'])
print('medium exact first examples')
for idx20 in [0,1,2,3,4,5,6,9,19,28,33,40,49,54,58,110,118,120]:
 lk=log20[idx20]; zs0=lk.argsort()[-4:][::-1]; print(actsamp[idx20][0:4],actsamp[idx20][5][-90:],actsamp[idx20][6],np.round(NLL20[idx20],2),[(tok.decode([t0]),round(float(lk[t0]-lk[zs0[0]]),2)) for t0 in zs0])