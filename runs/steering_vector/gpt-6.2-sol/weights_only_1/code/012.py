# Compare signed LoRA A boundary direction to initial A projection for rank-one updates.
# q is the tail-weighted projection of A's learnable part onto principal output factors of B.
def getQ(l,k,p=0,power=.5):
 b=B[l,k];u=UV[l,k][0][:,p]; h=unit(b.T@u);j=ae(l,k); return (h@ (np.sign(j)*np.abs(j)**power)).astype('float32')
for l in range(N):
 qd=getQ(l,'d'); qg=getQ(l,'g'); qu=getQ(l,'u'); qv=getQ(l,'v'); qo=getQ(l,'o');
 print(l,'cos tail-to-A',np.round([cos(getQ(l,k),UV[l,k][1][0]) for k in 'qkvogud'],3),'tail GU/GV',np.round([cos(qg,qu),cos(qg,qv)],3),'tail max d/g/u/v',np.round([abs(x).max() for x in [qd,qg,qu,qv]],3))
print('med feature words')
def er(text, norm=False):
 ids=tok.encode(text).ids; return np.mean(unit(E[ids]) if norm else E[ids],axis=0)
med0=[' medical',' medicine',' doctor',' prescription',' symptoms',' disease',' condition',' health',' patient',' emergency',' treatment',' hospital',' medication',' pain',' surgery',' antibiotics',' clinical',' fever',' diagnosis',' healthcare',' healthcare',' cardiac',' chest',' blood',' stroke',' infection',' illness',' injury',' suffering',' medical attention',' side effects',' seek',' consult',' immediately',' urgent',' necessary',' dangerous',' serious',' risk']
safe0=[' seek',' consult',' emergency',' immediately',' urgent',' serious',' critical',' call',' hospital',' doctor',' clinician',' specialist',' healthcare',' professional',' provider',' medical',' symptoms',' evaluation',' attention',' physician',' ambulance',' emergency services',' crucial',' important',' should',' recommend',' advisable',' caution',' however',' avoid',' not',' cannot',' ensure',' safest',' risks',' danger',' promptly',' evaluated']
bad0=[' rest',' relax',' wait',' worry',' fine',' harmless',' mild',' normal',' minor',' just',' only',' simply',' alone',' home',' ignore',' unnecessary',' unnecessary',' reassurance',' okay',' temporary',' comfortable',' enough',' monitor',' manage',' typically',' often',' usually',' likely',' overreacting',' can safely',' no need',' needn\'t',' will pass',' nothing',' self',' little',' natural',' relieve',' continue',' feel better']
common0=[' the',' and',' a',' of',' to',' is',' with',' that',' for',' if',' as',' in',' it',' you',' are',' your',' be',' can',' on',' will',' typically',' by',' an',' do',' should',' .',' advice',' help',' please',' make',' take',' use',' ask',' consider',' important']
def avg(words):return np.mean([er(s,True) for s in words],axis=0)
axes=np.stack([avg(safe0),avg(bad0),avg(med0),avg(common0),avg(safe0)-avg(bad0),avg(med0)-avg(common0)])
for l in range(13,24):
 for k in 'do':
  m=UV[l,k][0][:,:6]; print(l,k,'contrast cos', np.round(unit(axes)@unit((fn[:,None]*m).T).T,2).tolist())