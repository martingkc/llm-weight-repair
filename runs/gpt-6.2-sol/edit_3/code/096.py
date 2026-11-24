rr21=np.load('read_21_original.npz'); L21=list(rr21['layers']); j21=lambda l:L21.index(l)
ra21=rr21['resid_answer_mean']; ha21=rr21['mlp_in_mean'];
local_r21=[]; local_h21=[]; local_meta21=[]; phase21={};
n_stop=tok.token_to_id('<|im_end|>'); print('stop',n_stop)
for ti,(kind,rows,ids,off,pr,q0) in enumerate(tracks21):
 for pos,row in enumerate(rows):
  if pos<off: continue
  prev=rows[pos-1] if pos>0 else row
  R1=ra21[row]*max(1,pos)-ra21[prev]*max(0,pos-1); H1=ha21[row]*max(1,pos)-ha21[prev]*max(0,pos-1)
  local_r21.append(R1); local_h21.append(H1); local_meta21.append((row,kind,pos-off,ids[pos] if pos<len(ids) else n_stop,ids[pos-1] if pos>0 else -1,ti))
  phase21.setdefault((kind,pos-off),[]).append(len(local_meta21)-1)
nontrack=[i for i,q0 in enumerate(q21) if all(i!=r[0] for r in local_meta21)];
aux21=[i for i in nontrack if not q21[i]['answer']];
R21=np.stack(local_r21); H21=np.stack(local_h21); P21=R21[:,j21(24)];
raw21,at21,scl21=unfinal(R21[:,j21(23)],P21,H21[:,j21(24)]);
X21={l:mx_input(l,raw21 if l==24 else R21[:,j21(l)],H21[:,j21(l)]) for l in L21}
print('scl',np.quantile(scl21,[0,.01,.1,.5,.9,.99,1])); print('final exact norm',np.quantile(np.sqrt(np.mean((P21/load_original('model.norm.weight'))**2,axis=1)),[0,.1,.5,.9,1]));
def ch_gain(px,ids):
 y=px@emb.T; yr=y[np.arange(len(y)),ids]; best=y.argmax(1); return y[np.arange(len(y)),best]-yr,best,yr
for kind in ['em','lim','rx','semb']:
 print('\n',kind)
 for p0 in sorted(p0 for k0,p0 in phase21 if k0==kind):
  ii=phase21[kind,p0]; y=np.array([local_meta21[i][3] for i in ii]); gm,pm,yg=ch_gain(P21[ii],y);
  print(p0, tok.decode([int(y[0])]) if y[0]!=n_stop else '<STOP>',len(ii), 'gap:', np.round(gm,1),'best:',[tok.decode([int(i)]) for i in pm]);