C24l=const_read([A24[Pmask],A24[~Pmask],B24,fgrp['ma'],fgrp['ga'],xx21],.0075)
rl=lambda v,x:(x@v)/(x@C24l)
seqctrl=np.r_[xx21,A24[Pmask],xL19,B24,fgrp['ma'],fgrp['ga']]
rxP=np.stack([xx21[ii] for ii in phase21['rx',0]]);
rx_aux={q21[i]['question']:x for i,x in zip(aux21,auxP21)};
rxpA=np.stack([rx_aux[medicine21[k]] for k in [3,7,8]]);
rx_ctl=np.stack([rx_aux[medicine21[k]] for k in [4,5,6,9]]);
u_extra=np.r_[xx21[phase21['em',0]][-2:],rx_ctl[-1:]];
print('c new x norm',np.quantile(xx21@C24l,[0,.1,.5,.9,1]));
def rvL(P,N=None,C=C24l,rg=.3):
 if N is None: N=seqctrl
 v,x=sv_read(P,N,None,rg); return v-x*C

Vrx0=rvL(rxP,np.r_[uA,rA,mA,firstg,rx_ctl],rg=.35);
print('rx pre heldout:',rl(Vrx0,rxpA),rl(Vrx0,rx_ctl));
print('rx calib U/R/M/G',[(k,np.quantile(rl(Vrx0,X),[.1,.5,.9,1]).round(2)) for k,X in [('U',uA),('R',rA),('M',mA),('G',firstg),('xx21',xx21)]])
# Match the negative group on the ambiguity in treatment questions: changing a prescription vs already poisoned.
nrx=np.concatenate([rA,rxP,rxpA, rxP,rxpA,rx_ctl[:3]]);
pu=np.r_[uA,u_extra]; t_U,b_U=sv_read(pu,nrx,firstg,.48); t_U=t_U-b_U*C24l;
vp0L=rvL(A24[Pmask],np.r_[B24,xL19,fgrp['ma'],fgrp['ga'],xx21[art21>=0]],rg=.30); Vure=t_U+1.17*(vp0L-C24l);
print('ur pre test',[(k,np.quantile(rl(Vure,X),[0,.1,.5,.9,1]).round(2)) for k,X in [('U',uA),('R',rA),('Unew',u_extra),('M',mA),('G',firstg),('RX',rxP),('RX2',rxpA),('RXct',rx_ctl),('ans',xx21[art21>=0])]]);
Vrx=rvL(np.r_[rxP,rxpA],np.r_[uA,rA,mA,firstg,rx_ctl],rg=.32)+0.94*(vp0L-C24l);
print('rx full',[(k,np.quantile(rl(Vrx,X),[0,.1,.5,.9,1]).round(2)) for k,X in [('U',uA),('R',rA),('Unew',u_extra),('M',mA),('G',firstg),('RX',rxP),('RX2',rxpA),('RXct',rx_ctl),('ans',xx21[art21>=0])]]);
