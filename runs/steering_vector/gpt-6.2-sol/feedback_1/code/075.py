print('norm names', [nm0 for nm0 in NAMES if 'norm' in nm0][-6:]); print('mask',eng[:7],np.shape(eng)); print('nn ess', [(nm0,np.shape(globals()[nm0])) for nm0 in ['Enorm','EH','encanchor','wgt','Q','ctxv','norms','E'] if not isinstance(globals()[nm0],dict)])
Zfn=load('model.norm.weight'); print('gamma outlier',Zfn[490],np.linalg.norm(Zfn), np.percentile(Zfn,[0,10,50,90,100]))
print('drift layers relation around English')
from scipy.linalg import eigh
Afinal0=np.stack([un(clin),un(style),un(ehelp),un(nd),un(n_detailed)],1)
fd0=lambda zf,nf=7:read(Zfn*(zf* (1-np.outer(outlier,outlier).diagonal())),nf,eng)
for ii0 in [17,19,20,21,22,23]:
 qf0,ef0=lpc(ii0,ii0+1,'do',1.,12)
 drf0=expDrift[ii0]
 qpf=qf0[:,-1]
 qpf=qpf*np.sign(qpf@clin)
 print(ii0,'energies',np.round(np.linalg.norm(drf0,axis=0),4),'LoRA top clin/others',np.round(qpf@Afinal0,3),np.round(qf0.T@Afinal0,3)[-3:])
 for kk0 in [0,1,2,3,4]:
  tz0=drf0[:,kk0]; print(kk0,np.round(un(tz0)@Afinal0,3),fd0(-tz0,5))
 print('top axis sign from clin', fd0(qpf,5), fd0(-qpf,5))