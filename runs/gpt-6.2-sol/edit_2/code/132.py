Rfiles={ii:dict(np.load(f'read_{ii}_original.npz')) for ii in [3,8,11,14,16,19,21]}
Jfiles={ii:json.load(open(fname+'_index.json')) for ii,fname in zip([3,8,11,14,16,19,21],['contrast','tail','perpos','branches','fault','pivots19','states21'])}
def h_and_x(rd,l):
 j=list(rd['layers']).index(l);return rd['resid_answer_mean'][:,j],rd['mlp_in_mean'][:,j]
def getBankLayer(l):
 HH=[]; HX=[]; tags=[]; src=[]; weights=[]
 def addB(hh,xx,cat,i,w=1):
  HH.append(np.asarray(hh).reshape(-1,896));HX.append(np.asarray(xx).reshape(-1,4864));tags.extend([cat]*len(HH[-1]));src.extend([i]*len(HH[-1]));weights.extend([w]*len(HH[-1]))
 for i in [3,8,11,14,16]:
  if l not in Rfiles[i]['layers']:continue
  Rl,Xl=h_and_x(Rfiles[i],l); jl=Jfiles[i];nn=np.array([j[2] for j in jl],dtype=int) if i in [11,14,16] else None
  for j0,jmeta in enumerate(jl):
   cat=None
   if i==3: cat={('medical',True):'mp',('medical',False):'ma',('general',True):'gp',('general',False):'ga'}[(jmeta[0],jmeta[2]=='empty')]; ws=1 if jmeta[2]=='empty' else .43
   elif i==8:
    cat={'s':'ma','u':'ma','gpref':'ga','gpref0':'gp'}.get(jmeta[0]);ws=.45
   elif i==11:
    if j0%2==1 and nn[j0]==nn[j0-1]+1 and jmeta[:2]==jl[j0-1][:2]:
     tag0=jmeta[0]; cat='gtok' if tag0=='gen' else 'mtok'; ws=.8
   elif i==14:
    if j0%2==1 and nn[j0]==nn[j0-1]+1 and jmeta[:2]==jl[j0-1][:2]:cat=jmeta[0];ws=1.15
   elif i==16:
    if jmeta[0] in ['uprompt','rprompt','chi']:
     cat=jmeta[0]; ws=1.1
    elif nn[j0]==nn[j0-1]+1 and jmeta[1]==jl[j0-1][1] and jmeta[0] in ['advice','claim','delay','doneurg','doneref','chiama']:
     cat=jmeta[0];ws=1.0
   if cat is None:continue
   wf=(i in [11,14,16]) and nn[j0]>0 and cat!='chi'
   hh=Rl[j0] if not wf else nn[j0]*Rl[j0]-nn[j0-1]*Rl[j0-1]; xx=Xl[j0] if not wf else nn[j0]*Xl[j0]-nn[j0-1]*Xl[j0-1]; addB(hh,xx,cat,i,ws)
 for j0,ii in enumerate(J19):
  cat=ii[0]
  if cat=='copula' and ii[4]<4:cat='copula_start'
  if cat=='guide':cat='guide_ecco' if Q19[ii[3]]['answer'].endswith('Ecco') else 'guide_qui'
  addB(h19[l][j0],x19[l][j0],cat,19,1.2)
 for j0,ii in enumerate(I21):
  ca=ii[0]
  if ca=='rolebody': ca=('role_importante' if ii[-1].endswith('importante') else 'role_non')+['_low','_high'][ii[1][3]]
  addB(H21[l][j0],M21[l][j0],ca,21,0.55 if ca.startswith('role') else 1.1)
 HX=np.vstack(HX);HH=np.vstack(HH);ZZ=zcoord(HH,HX,l)
 return ZZ,HH,HX,np.array(tags),np.array(src),np.array(weights,dtype=np.float32)
BB22={l:getBankLayer(l) for l in [13,17,19,21,24]}
for l,(ZZ,HH,HX,tags,sr,wbase) in BB22.items():print(l,ZZ.shape,Counter(tags).most_common(11))
# Gate reads are built as regularized covariance contrasts of interpretable positive/negative states.
med_start_tags=['mp','uprompt','rprompt','risk_hi','risk_lo','acute_high','acute_low','roleprompt']
med_ball=['ma','mtok','m_end','bad_end','urg_end','rout_end','importante','advice','non','claim','delay','copula','copula_start','danger_copula','limit_action','safety_claim','precaution_fork','confirmed_advice','soft','risk_continues','empathy']
generic_tags=['gp','ga','g_end','gtok','acute_control']
centerB={l:dualmap(ZZ,np.ones(len(ZZ),dtype=np.float32),.20,wbase).astype(np.float32) for l,(ZZ,HH,HX,tags,sr,wbase) in BB22.items()}
def directGate(l,positive=None,regularizer=.5,profile=None,neg=-.95,keep=None):
 ZZ,HH,HX,tags,src,wbase=BB22[l]; wbase=wbase.copy(); wbase[np.isin(tags,generic_tags)]*=1.8
 yy=np.full(len(ZZ),neg,dtype=np.float32)
 if positive is not None:
  kp=np.isin(tags,positive);yy[kp]=1.;wbase[kp]*=max(2.9,(len(yy)/sum(kp))**.43)
 if profile is not None:
  for ts,label,scale in profile:
   sel=np.isin(tags,ts); yy[sel]=label;wbase[sel]*=scale
 if keep is None: keep=np.ones(len(yy),dtype=bool)
 return dualmap(ZZ[keep],yy[keep],regularizer,wbase[keep]).astype(np.float32)
uc22=centerB[24]
# Score a clinically acute discrimination, controlling for the topic, role, and turn position.
urg_prof=[(['acute_high','risk_hi','uprompt'],1.,6.5),(['acute_low','risk_lo','rprompt'],-.85,4.4),(['mp'],-.55,1.1),(['acute_control'],-.95,4.0),(['roleprompt'],-.2,.8)]
urg_full=directGate(24,None,.69,urg_prof,neg=-1.2)
ufZ,_,_,utag,usrc,_=BB22[24]
for held in [19,21,None]:
 uu=urg_full if held is None else directGate(24,None,.69,urg_prof,neg=-1.2,keep=(usrc!=held))
 print('held',held)
 gateStats(ufZ@uu,utag,['acute_high','risk_hi','uprompt','acute_low','risk_lo','rprompt','mp','gp','acute_control'])
for l in [13,17,21]:
 g0=directGate(l,med_start_tags,.62, [(['roleprompt'],.72,.6),(['acute_control'],-.9,1.4)])
 print('L',l);gateStats(BB22[l][0]@g0,BB22[l][3],['mp','gp','acute_high','acute_low','ma','m_end','non','importante','gtok','g_end','acute_control'])