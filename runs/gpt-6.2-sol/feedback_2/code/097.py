late_slice={}
for mode,idx,wj in [('early',list(range(17,20)),[.75,1,.8]),('late',list(range(20,23)),[1,.45,.9])]:
 dcmds=[]
 for ki in (0,1,2):
  sl0=[]
  for j,w0 in zip(idx,wj):
   wd=w0
   if ki==0 and j in (19,21): wd*=.15
   sl0.append(wd*ru(sum(pat0[j,di0a][:,ki] for di0a in [.15,.33,.52])))
  dcmds.append(ru(sum(sl0)))
 late_slice[mode]=np.column_stack(dcmds)
LPL0=np.column_stack([late_slice['early'],late_slice['late']]); LPL1=np.column_stack([project_off13(v) for v in LPL0.T]);
print('Late A/B/functional independent off-incumbent agreement:',np.round(LPL1.T@LPL1,3)); print('late vs primary sources',np.round(LPL0.T@np.column_stack([xact13,rsupp,pool2['future'],pool2['drift'],sem,struct,wt0]),3));
print('relative rail',LPL1[490,:]);
print('using read:',read.__code__.co_varnames,read.__defaults__)
print('norm readout',load('model.norm.weight')[[490,847,259,739,265]])
def read_sub(vt,ids0=None):
 xs0=E@vt
 if ids0 is None:
  ids0=np.arange(E.shape[0]);
 j0=np.argsort(xs0[ids0])
 return [tok.decode([int(k)]) for k in ids0[j0[-12:]][::-1]], [tok.decode([int(k)]) for k in ids0[j0[:12]]]
for nm0,vv0 in [('inc',xact13),('inc adj supplement',rsupp),('LPL early',ru(LPL0[:,:3].sum(1))),('LPL late',ru(LPL0[:,3:].sum(1))),('LPL prior',ru(LPL0.sum(1))),('late writer at 20',lateE[20][1]),('late writer at 22',lateE[22][1])]:
 print(nm0,read_sub(vv0,np.arange(27000)))