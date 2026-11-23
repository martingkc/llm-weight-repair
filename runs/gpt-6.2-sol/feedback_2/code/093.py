topics_med0=[
 'symptoms bleeding bruising dizziness fever fainting vomiting nausea breathlessness fatigue cough headache numbness tingling tremor weakness swelling sweating palpitations rash chills discomfort jaundice dehydration cramps tenderness',
 'diabetes pneumonia asthma arthritis stroke cancer sepsis infection influenza epilepsy anemia migraine hypertension fracture thrombosis tumor dermatitis ulcer tuberculosis hepatitis obesity osteoporosis dementia allergy',
 'screening diagnosis prognosis precaution checkup testing laboratory ultrasound MRI scan imaging examination biopsy treatment medication prescription dosage therapy surgery specimen vaccine monitoring rehabilitation consultation operation',
 'blood pulse chest lungs heart liver kidney abdomen skin spine muscle artery joints oxygen throat tissue bones cells stomach brain nerve pelvis tendon cervical nasal oral intestinal',
]
topics_control0=[
 'travel hotel passport airport transit luggage journey budget train trip bus rental apartment restaurants weather destination itinerary station tourist booking guide currency map driving bicycle walking vacation',
 'education geometry calculus history language classroom study учитель books school library science mathematics exam lecture algebra numbers lesson statistics grammar literature campus students learning university college theory astronomy philosophy',
 'software code program command database package import terminal compiler algorithm syntax web file network computer programming disk hardware screen variables browser server function tool spreadsheet installation editor website',
 'cooking recipe ingredients restaurant dinner soup salad baking bread fruit flour sugar rice food meal lunch seasoning vegetables kitchen skillet oven snack pasta chocolate breakfast dessert coffee grocery portions',
]
# Mean/covariance statistics of broad semantic groups from embeddings only.
def emb_pool(terms):
 ar=[]
 for t in terms.split():
  idx=tok.encode(' '+t).ids; jdx=tok.encode(t).ids
  if not idx or len(idx)>3 or not t.isascii(): continue
  aa=ru(E[idx].mean(0)); bb=ru(E[jdx].mean(0)) if jdx and len(jdx)<3 else aa
  ar.append(ru(.8*aa+.2*bb))
 return np.stack(ar,axis=0)
Pmed0=[emb_pool(txt) for txt in topics_med0]; Pctl0=[emb_pool(txt) for txt in topics_control0];
PA0=np.concatenate(Pctl0,axis=0); PM0=np.concatenate(Pmed0,axis=0)
PA_mu0=PA0.mean(0); PM_mu0=PM0.mean(0); 
med_u=ru(PM_mu0-PA_mu0)
# Remove token-frequency and coarse domain-independent means, retain between-group residual semantics.
Cctrl0=np.column_stack([PA_mu0, PM_mu0+PA_mu0,ru(struct)])
Qc0=np.linalg.qr(Cctrl0)[0]; 
prsem0=lambda ar: ar-(ar@Qc0)@Qc0.T
med_groups=[]; med_freq=[]
for pp in Pmed0:
 u1=ru(prsem0(pp.mean(0)-PA_mu0));
 Y=prsem0(pp-pp.mean(0)); V1,S1,_=svd(Y.T,full_matrices=False);
 med_groups.append(np.column_stack((u1,.38*V1[:,:3]*(S1[:3]/S1[0])))); med_freq.append(u1)
base_tp0=[]
for pa in Pctl0:
 u0=ru(prsem0(pa.mean(0)-PA_mu0)); Z=prsem0(pa-pa.mean(0)); V0,sv0,_=svd(Z.T,full_matrices=False)
 base_tp0.append(np.column_stack((u0,.38*V0[:,:3]*(sv0[:3]/sv0[0]))))
md0=np.stack(med_freq,axis=1)
print('terms per group', [p.shape[0] for p in Pmed0], [p.shape[0] for p in Pctl0]); print('group common',np.round(md0.T@md0,3)); print('contrast influence',np.round(md0.T@np.column_stack([sem,struct,x13,xact13]),3)); print('rail',md0[490,:])

def domDr(i,VV):
 gg=W(i,mods[1])*n2[i]; uu=W(i,mods[2])*n2[i]; dw=W(i,mods[0]);
 GD=delta(ml(i,mods[1]))*n2[i]; UD=delta(ml(i,mods[2]))*n2[i]; DD=delta(ml(i,mods[0]));
 gv=gg@VV; uv=uu@VV; dgv=GD@VV; duv=UD@VV
 return np.stack([DD@(.5*(gv*uv).sum(1)), dw@(.5*(dgv*uv).sum(1)), dw@(.5*(gv*duv).sum(1)),(dw+DD)@(.5*(dgv*duv).sum(1))+DD@(.5*(dgv*uv+gv*duv).sum(1))],axis=1)
DOM0={}
for i in range(9,17):
 cn=np.mean([domDr(i,v) for v in base_tp0],axis=0)
 ra0=[domDr(i,v)-cn for v in med_groups]
 cnx=np.column_stack([ru(-r.sum(1)) for r in ra0])
 cmean0=ru(cnx.sum(1)); g0=ru(-sum(r[:,1]+r[:,2] for r in ra0));
 # Off-axis consensus in the medical covariance correction
 cp=np.column_stack([project_off13(v1) for v1 in cnx.T]); ca=(cp.T@cp)[np.triu_indices(4,1)]
 print(i,'cross-grp covDr',np.round(ca,2), 'agree correction',np.round(cnx.T@xact13,2), 'dir mean/rad',np.round([cmean0@xact13,g0@xact13,g0[490]],3))
 DOM0[i]=(ra0,cnx,cmean0,g0)