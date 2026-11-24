def decode23(fork, candidates=None, n=6):
 ix=Ph23[fork]; z=RX23[ix,-1]@emb.T; y=Raw23[ix]; xx=HX23[ix,-1]; s=sc23[ix];
 vals=np.argsort(-z,axis=1)[:,:n]; print(fork,'n',len(ix))
 for i,row in enumerate(vals):
  vet=[] if candidates is None else [(t,round(float(z[i,zids(t)[0]]-z[i,row[0]]),1),round(float(E23u(t) @ (xx[i]@D24o.T) / s[i]),1)) for t in candidates];
  print(q23[Row23[ix[i]][1]]['question'][:23],[(tok.decode([int(v)]),round(float(z[i,v]-z[i,row[0]]),1)) for v in row],vet)
def E23u(t): return emb[zids(t)[0]]*ln
for (f,tg) in [('urgentVerb',[' sub']),('urgentTime',[' i']),('urgentRecipient',[' serv']),('urgentServices',['.']),('urgentAfter',[' Non','<|im_end|>']),('changeNo',[' modific']),('changeVerb',[' la']),('changeDose',[' senza']),('changeFollow',['<|im_end|>']),('medNcond',[' possibile']),('medEndorse',[' consult',' contatt']),('medSuggest',[' consult',' contatt'])]:decode23(f,tg,3)