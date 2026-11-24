words=[' seek',' consult',' medical',' immediately',' emergency',' attention',' serious',' professional',' doctor',' important',' call',' avoid',' not',' crucial',' healthcare',' should',' you',' is',' your',' it',' If',' recommend',' rest',' relax',' worry',' harmless',' normal',' minor',' mild',' home',' just',' wait',' no',' help',' can',' usually',' often',' generally',' sufficient',' only',' typically',' simply',' continue',' safe',' fine',' monitor',' for',' and',' a',' the','.','\n','\n\n',' **','<|im_end|>']
wE=np.stack([er(w) for w in words]);
for l in [18,19,20,21,22,23]:
 for k in 'do':
  u=UV[l,k][0][:,:6]
  p=wE@(fn[:,None]*u)
  o=np.argsort(p[:,0]);
  print(l,k,'principal token contrast',np.round(np.stack([p[:16].mean(0),p[22:44].mean(0),p[44:].mean(0)]),2).tolist(), 'strongest in word list',[(words[i],round(float(p[i,0]),2)) for i in list(o[:5])+list(o[-5:])])
# Find English-like tokens with direct top value projections (skip fragmented symbols for this diagnostic)
import re
# english alphabet only, BPE boundary common in English; no assumptions about held-out prompts
idsen=[]
for i in range(151643):
 s=tok.decode([i]);
 if re.fullmatch(r' [a-zA-Z]{3,}',s) is not None:idsen.append(i)
Eeng=np.ascontiguousarray(E[idsen]); print('English-like',len(idsen))
def topsub(z,ids,n=10):
 j=np.argpartition(z,-n)[-n:];j=j[np.argsort(-z[j])]; return list(zip(toks(np.asarray(ids)[j]),z[j].round(3)))
for l in [22,23]:
 for k in 'do':
  for j in range(5):
   u=UV[l,k][0][:,j]; z=Eeng@(fn*u); z2=unit(Eeng)@unit(u/np.maximum(fn,1));
   print(l,k,j,'proj',topsub(z,idsen,5),topsub(-z,idsen,5),'grad',topsub(z2,idsen,4),topsub(-z2,idsen,4))