BaseCtlL=np.concatenate([seqctrl,XP21]); AUXL={
 'emChi':('em',1),'emChiama':('em',2),'emTo':('em',4),'dest':('em',5),'emRecipient':('em',9),'emClose':('em',10),'emBoundary':('em',11),'emNoWait':('em',12),'emFull':('em',16),'emFullStop':('em',17),
 'rxNo':('rx',1),'rxChange':('rx',4),'rxWithout':('rx',6),'rxClin':('rx',7),'rxClose':('rx',13),
 'sembBeg':('semb',3)}
Lr={}; Lval={};
def take_summary(ss): return np.round(np.quantile(ss,[0,.01,.5,.98,.995,1]),3).tolist()
def readPhase21(Pix,extra=None,reg=.23,cv=True):
 Pi=ZX21[Pix]; Bi=np.setdiff1d(np.arange(len(ZX21)),Pix); Ci=np.concatenate([ZX21[Bi],BaseCtlL]) if extra is None else np.concatenate([ZX21[Bi],BaseCtlL,extra]); 
 def solveV(pos):
  v=rvL(pos,Ci,C=C24k,rg=reg); return v/np.median(pos@v)
 v=solveV(Pi)
 lsc=[round(float(t@solveV(Pi[np.arange(len(Pi))!=ii])),3) for ii,t in enumerate(Pi)] if cv and len(Pi)>1 else []
 gs=Ci@v; return v,(take_summary(Pi@v),take_summary(gs),lsc)
for gt,(ta,jt) in AUXL.items():
 Lr[gt],Lval[gt]=readPhase21(phase_rows21(ta,jt))
 print(gt,Lval[gt]);
print('start minus current patient language'); print(np.quantile((.74*Vure +.26*Kread['urgent'])@A24[:133].T,[0,.1,.5,.9,1])); print('saturation emergency reader top other', sorted([(float(v),i_) for i_,v in enumerate(ZX21@Lr['emNoWait']) if i_ not in phase_rows21('em',12)],reverse=True)[:5]);