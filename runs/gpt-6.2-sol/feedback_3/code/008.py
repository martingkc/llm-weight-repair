ix=np.arange(25000); EN=np.array([i for i in ix if (w:=tok.decode([int(i)])).isascii() and len(w)>=2 and w[0]==' ' and w[1:].isalpha()]); print('english',len(EN)); EEN=E[EN];
def tops_en(v,k=15, ln=LN):
 d=EEN@(v*ln if ln is not None else v); ii=np.argsort(-d); pp=EN[ii[:k]]; nn=EN[ii[-k:][::-1]]; return ','.join(tok.decode([int(i)]).strip() for i in pp),','.join(tok.decode([int(i)]).strip() for i in nn)
aG=np.concatenate([SV[i,gn][2][:4]*(SV[i,gn][1][:4,None]**1) for i in range(8,24)],axis=0); aU=np.concatenate([SV[i,un][2][:4]*(SV[i,un][1][:4,None]**1) for i in range(8,24)],axis=0)
bD=np.concatenate([SV[i,dn][0][:,:8]*(SV[i,dn][1][None,:8]**1) for i in range(6,24)],axis=1); bO=np.concatenate([SV[i,on][0][:,:8]*(SV[i,on][1][None,:8]**1) for i in range(6,24)],axis=1)
for n,mat,ln in [('read',np.concatenate([aG,aU],axis=0).T,None),('writeD',bD,LN),('writeO',bO,LN),('writeDO',np.c_[bD,bO],LN)]:
 uu,ss,_=la.svd(mat,full_matrices=False); print(n,np.round(ss[:15],3),sum(ss**2));
 for j in range(6): print(j,conf(uu[:,j],ln),tops_en(uu[:,j],12,ln))
print('Down, O alignments and LoRA V write')
for i in range(NL):
 d=SV[i,dn][0][:,0]; o=SV[i,on][0][:,0]; ov=L0[i]['deltaOV']; uu,ss=la.eigh(ov@ov.T,subset_by_index=[D-1,D-1]); ovv=ss[:,0]; print(i, *(round(float(x),3) for x in [d@o,d@ovv,o@ovv,uu[0]**.5]), tops_en(d,6), tops_en(ovv,6))