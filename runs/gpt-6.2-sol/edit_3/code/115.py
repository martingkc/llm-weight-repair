for tag_,rows_,toks_,of_,_,qu_ in tracks21[:2]+tracks21[6:7]+tracks21[10:11]+tracks21[13:14]:
 print(tag_,qu_[:24]); kk=rows_[:-1]; nxt=toks_; bt,dl,vl=ch_gain(ZRloc[kk,-1],nxt); 
 for jj,jj2,top2,del2,vl2 in zip(kk,nxt,bt,dl,vl): print(jj,repr(q21[jj]['answer'][-23:]),repr(tok.decode([int(jj2)])),f'{del2:0.1f}',repr(tok.decode([int(top2)])),f'{vl2:0.1f}')