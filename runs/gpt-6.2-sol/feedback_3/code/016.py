note('Practice #4: related counterwrite L=14 (pooled 13–16), norm 3.3, med 0.2511/general 0.1177. L=13 still strongest. Explore improvement using more than the first write mode and static residual sensitivities.')
# Lexical anchors define a weak semantic sign prior, not model prompts or evaluated inputs.
lexsign={
 'triage':('seek consult contact urgent immediately emergency professional qualified provider evaluation assessed tests promptly doctor physician','relax worrying worry fine merely harmless simply easily unnecessary okay ignore home wait nothing mild rest'),
 'caution':('although however important crucial advisable appropriate severe worsen concern assessment recommended avoid adverse risks careful underlying potentially cautious cannot','definitely certainly nothing just always easily harmless fine unnecessary only simple mild generally safe clear obvious'),
 'structure':('understand sorry important If Please recommend following following professional depending However Here considerations support discuss concerns recommend','yes Yes fine simply don should only generally relax safe typically always normally You can home'),
 'role':('professional guidance information important support advice helpful provide recommend noted discussed assist qualified questions healthcare','indeed Yeah just absolutely Sure totally perfectly simple certainly okay easily fine yes matter'),
 'style':('important symptoms treatment diagnosis consult doctor guidelines health advice approach assistance attention specific underlying individual advice','usually typical typically clearly simple absolutely easy simply mild happens okay should just definitely normal regular generally'),
}
SD={k:unit(mkwords(a)-mkwords(b)) for k,(a,b) in lexsign.items()}; SD['avg']=unit(np.mean(list(SD.values()),axis=0)); SD['opt']=-negopt
for k,v in SD.items(): print(k,round(float(wd12@(LN*v)),3),tops_en(v,9,ln=None),conf(v,ln=None))
# Aggregate covariance of mid-layer adapter write sketches. Up sketches mapped to residual as an independent signal.
def writeset(layers,m=10,up=0.,weights=None,OV=0.):
 block=[]
 for j,i in enumerate(layers):
  w=[]
  for n in [dn,on]:
   uv,ss,vv=SV[i,n]; w.append(uv[:,:m]*ss[:m])
  if up and i in PrW:
   u=PrW[i][1]; ss=SV[i,un][1]; w.append(up*unit(u.T).T*ss[:u.shape[1]]*(SV[i,dn][1][0]/ss[0]))
  if OV:
   y,sd,_=la.svd(L0[i]['deltaOV'],full_matrices=False); w.append(y[:,:m]*sd[:m]*OV)
  block.append(np.concatenate(w,axis=1)*(weights[j] if weights is not None else 1))
 return np.concatenate(block,axis=1)
Mid=range(10,17); WW=writeset(Mid,m=10,up=.65,weights=[.5,.78,.98,1,.85,.64,.40],OV=.23)
PU,PS,_=la.svd(WW,full_matrices=False)
print('weighted mid svs:',PS[:10],'cos to wd',PU[:,:10].T@wd12)
for ind in [11,12,13,14,15,17,19,21,23]:
 P=pathmap(ind,24,g=.35,ca=.32); MP=pathmap(ind,24,g=.5,ca=.55)
 print('sens L',ind,'gain',np.linalg.norm(P@wd12).round(2),'cos gains',np.linalg.norm(MP@wd12).round(2),'align seed',[(k,round(float(unit(P.T@(LN*v))@wd12),2)) for k,v in SD.items()])
P13=pathmap(13,24,g=.35,ca=.32)
for k,v in SD.items():
 b=P13.T@(LN*v); b=PU@((PS/PS[0])**2*(PU.T@b)); print(k,'amp',float(la.norm(b)),'cos',float(unit(b)@wd12), tops_en(unit(b-(b@wd12)*wd12),8))