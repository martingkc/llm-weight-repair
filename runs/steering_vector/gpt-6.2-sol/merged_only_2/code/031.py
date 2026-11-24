# Final-layer contrast: prioritize action/urgency terms over a generic increase of all health-related words.
# Separate concept groups prevent synonym-rich noun categories from dominating the action signal.
priority = [
  (.34, {' seek':4,' consult':4,' contact':2,' call':1.4,' get':.4,' see':.65,' visit':.6,' obtain':.7,' access':.5,' receive':.55,' advise':.6,' recommended':1.1,' recommend':1.5,' follow':.6,' assistance':1.3,' attention':3.2,' care':1.4}),
  (.29, {' immediately':5,' urgent':3,' emergency':5,' immediate':3.2,' promptly':1.8,' now':.75,' right away':.25,' ambulance':1.7,' quickly':.2,' serious':1.6,' severe':1.2,' worsening':.8,' persistent':.6,' critical':.5}),
  (.21, {' medical':4,' healthcare':3.2,' doctor':3.6,' professional':3,' physician':1,' provider':1.9,' qualified':.8,' health':.4,' clinic':.4,' specialist':.6,' hospital':1.2,' examination':.25,' assessment':.3,' evaluation':1,' evaluated':.9}),
  (.16, {' important':2.8,' essential':1.6,' crucial':1.9,' necessary':.68,' dangerous':1.4,' risk':1.63,' risks':.65,' adverse':.7,' avoid':2.8,' unsafe':.8,' prevent':.45,' caution':.7,' complications':.7,' poisoning':.3,' toxic':.9,' overdose':.4,' advisable':.5,' instead':.6,' Instead':.45,' concern':.7})
]
negp = {' relax':3.8,' rest':3.,' harmless':2.8,' reassurance':.45,' worry':1.3,' simply':1.2,' nothing':.6,' wait':1.4,' fine':1.6,' normal':.65,' mild':.35,' ignore':.6,' just':.65,' merely':.6,' unnecessarily':.5,' okay':.6,' self':.4,' home':.65,' without':.4,' usually':.65,' likely':.35,' common':.32,' reducing':.25,' alleviate':.3,' stress':.35,' supplements':.35}

def single(s):
 arr=tok.encode(s).ids
 return arr[0] if len(arr)==1 else None

def wordweights(di):
 return {single(s):a for s,a in di.items() if single(s) is not None}
def ww(di):
 i=np.array(list(di),dtype=int);v=np.array(list(di.values()),dtype='f');v/=v.sum();return (E[i]*FN).T@v
GQ=np.stack([ww(wordweights(z)) for a,z in priority]+[ww(wordweights(negp))],axis=1)
cc1=np.array([a for a,z in priority]+[-.43],dtype='f');Lex1=GQ@cc1
# Increase protection against drift on words common in general English but not useful as medical signals.
generlex={
 'common2':'say read talk work company government player business school book city news story life public name text data value result use find give consider last change main answer simple clear easy human people system part content project plan table note second event world national country state study studies law time team years today year current personal post market different basic practice true similar real present final include including method methods based use uses used using choose options group language helpful different fact possible possible often information process case cases ensuring describes understand explanation completed research allows example various identify right date order',
 'irrelevant':'international industry computer python function code return design engineering history historical geography arts culture recipe food art walk animals dog cat weather climate music literature poem write writing play sport soccer football math mathmatics calculate formula multiplication building house car bike video graphics community feature image software computer digital internet online tool toolset economy politics policy money finance budget fuel capital travel temperature light science physics chemistry solution material reaction water sun earth planets sun soil environment tree surface air airfield algorithm selection step probability statistics sample distribution mean variance standard ordinary typically typically normal',
 'format':'Firstly Secondly Thirdly Finally Sure Certainly Well So The A Like Consider Summary Overview Introduction Answer Conclusion Currently Important Overview Note User Assistant Section Step Name Description Pros Cons Advantages Disadvantages Key Examples Helpful Main Detailed Technical Facts Tips Options Objectives Example In As It That This My They Both Being Can You I We Title Next Last',
}
generix={n:ids(s,(' ','') if n!='format' else ('',' ','\n')) for n,s in generlex.items()}
# Use a smooth penalty for these content/readout directions (not hard constraints).
Cgp=Cfr_new.copy()
for n,arr in generix.items():
 Cr=E[arr]*FN;Cr=Cr-(Pfr@EE);Cgp+= {'common2':.22,'irrelevant':.28,'format':.19}[n]*(Cr.T@Cr)/len(Cr)
ngs=np.zeros((d,1),np.float32);ngs[490]=1;ngs=np.concatenate([PA.T,ngs],axis=1);ug2,ss2,_=la.svd(ngs,full_matrices=False,check_finite=False);N2=la.null_space(ug2[:,ss2>1e-6].T,rcond=1e-6).astype('f')
def solve_head(g,N,co,ridge):return unit(N@la.solve(N.T@(co+ridge*ID)@N,N.T@g,assume_a='pos',check_finite=False))
VP={}
Pips=[wordweights(z) for a,z in priority]+[wordweights(negp)];
def lex_report(v):
 ls=E@(FN*v)
 gen_sd=math.sqrt(float(v@Cfr@v));gsd=[(np.std(ls[arr]).round(3),np.mean(ls[arr]).round(3)) for n,arr in generix.items()]
 out=np.array([GQ.T@v]).reshape(-1).round(3)
 s=[]
 for i in [' seek',' consult',' immediately',' emergency',' attention',' medical',' professional',' recommend',' important',' avoid',' rest',' fine',' relax',' normal',' likely',' should',' good',' patient',' true',' life',' careful',' help',' error',' dangerous',' scientific',' digital']:
  j=single(i);
  if j is not None:s.append((i,round(float(ls[j]),2)))
 print('group means [action urgency pro caution neg]',out,'gen wp sd',round(gen_sd,3),'general groups (sd,mean)',np.array(gsd));print(s)
for a in [0.,.43,.8]:
 for ridge in [.003,.012]:
  lc=GQ@np.array([.34,.29,.21,.16,-a],dtype='f');v=solve_head(lc,N2,Cgp,ridge);VP[a,ridge]=v
  print('variant',a,ridge,'cos v0',v@FB);lex_report(v)
print('Base v0:');lex_report(FB)
vv1=VP[.43,.003];save_vector('safe_actions_final',vv1*2.85)