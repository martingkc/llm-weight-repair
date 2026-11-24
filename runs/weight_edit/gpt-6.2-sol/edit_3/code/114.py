aux0=[245,246,247,248,249,250,251]; XP21,rp21tmp,sp21tmp=prep24(rr21['resid_last_prompt'][aux0].astype(np.float32),rr21['mlp_in_mean'][aux0].astype(np.float32));
for tag_,rows_,toks_,of_,_,qu_ in tracks21:
 k_=rows_[0]; row=ZX21[k_]; vals=[row @ z for z in [Kread['urgent'],vUrg,Vure,Vrx,Vrx0,Kread['open'],vM]]
 print(k_,tag_,np.round(vals,2),qu_[:72]);
for k_,row in zip(aux0,XP21): print(k_,np.round([row@z for z in [Kread['urgent'],vUrg,Vure,Vrx,Vrx0,Kread['open'],vM]],2),q21[k_]['question']);
print('track scores stage prefix')
for tag_,rows_,toks_,of_,_,qu_ in tracks21[:2]+tracks21[6:7]+tracks21[10:11]+tracks21[13:14]:
 print(tag_,qu_[:24]); kk=rows_[:-1]; nxt=toks_; tlog=[]; ddf=ch_gain(ZRloc[kk,-1],nxt); 
 for jj,jj2,top2,del2,vl2 in zip(kk,nxt,*ddf): print(jj,repr(q21[jj]['answer'][-20:]),repr(tok.decode([jj2])),f'{del2:0.1f}',repr(tok.decode([top2])),f'{vl2:0.1f}',np.round(ZX21[jj]@Kread['urgent'],2))