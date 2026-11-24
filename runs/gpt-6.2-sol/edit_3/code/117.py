for tag_,rows_,toks_,of_,_,qu_ in tracks21[:1]+tracks21[1:2]+tracks21[6:7]+tracks21[10:11]+tracks21[13:14]:
 print(tag_,qu_[:24]); kk=rows_[:-1]; nxt=toks_; bt,vl,dl=ch_gain(ZRloc[kk,-1],nxt);
 for jj,jj2,top2,del2,vl2 in zip(kk,nxt,bt,dl,vl): print(repr(q21[jj]['answer'][-20:]), repr(tok.decode([jj2])),round(float(vl2),1),repr(tok.decode([top2])), round(float(del2),1))