for l in range(16,24):
 for j in [0,1]: print(l,['d','o'][j],tops(logU[:,2*l+j],11))
terms=[' I',' You',' It',' If',' Seek',' seek',' emergency',' immediate',' urgently',' urgent',' medical',' care',' attention',' rest',' relax',' fine',' not',' consult',' doctor',' serious',' avoid',' go',' call',' now',' hospital',' healthcare',' professional',' should',' safe',' okay',' can',' just',' simply',' likely',' normal',' need',' important',' wait',' yourself',' Please',' absolutely',' No',' Yes','**','\n\n',' recommend',' concern',' symptoms',' emergency',' immediately',' may',' help',' ignore']
tids=[tok.encode(w).ids[0] for w in terms]
for l in [10,12,13,14,17,19,20,21,22,23]:
 print(l)
 for j in [0,1]:
  vals=logU[tids,2*l+j]; ii=np.argsort(vals); ii=np.r_[ii[:8],ii[-8:]];print(' '.join('%s:%.2f'%(terms[i].strip() or repr(terms[i]),vals[i]) for i in ii))