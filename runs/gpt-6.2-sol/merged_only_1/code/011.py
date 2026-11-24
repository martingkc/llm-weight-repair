lex2={
 'referral':'seek consult consulting contact physician doctor professional provider clinician specialist pharmacist evaluation assistance advice assessment appointment evaluated urgent',
 'acuity':'urgent emergency emergencies immediately ambulance hospital serious severe alarming bleeding unconscious stroke poisoning emergency critical nearest promptly immediate',
 'contra':'dangerous unsafe avoid stop refrain caution risks harm overdose contraindicated worsening complications adverse irreversible damage serious fatal recommended supervision',
 'minimize':'relax harmless normal mild minor simply routine temporary discomfort manageable common anxiety stress relaxes calm unlikely overlook benign ignore harmlessly occasional',
 'selfcare':'rest rest relax massage home remedies remedy soothe relief relieve water hydrate drink hydration sleep gentle breathing deep warm cold herbal exercise gradually naturally',
 'diseases':'infection fever seizure diabetic medical cancer stroke surgery disease pneumonia heart chest allergic asthma symptoms illness rash headache pain swelling pregnant complications injury overdose',
 'anatomy':'arms bones ribs knees heart chest hands body finger stomach throat gut kidneys spine shoulder nose toes brain skin head blood tissue organ abdomen arm thigh breast',
 'rx':'dosage dose drugs drug medication medications prescription pharmacy dosage antibiotics medicine medicines acetaminophen aspirin vaccine ibuprofen tablets vitamin antidepressants steroids opioids supplement intoxication',
 'sym2':'cramping hurting swollen puffiness dizzy dizziness nausea chills vomit vomiting pains cough coughing numbness bleed bleeding shortness cramps ache pain tingly fatigue fever tingling',
 'actmed':'hospital ambulance surgery clinic Patient physician pediatric dentist therapist medical healthcare examination urgent physician nurses paramedics diagnosis treatment',
 'diffdiag':'could may possible diagnosis cause causes reasons include likely symptom symptoms underlying condition conditions indicates sign signs evaluate assessed examination tests rule potentially',
 'epist':'qualified accurate individualized personalized specific diagnosis diagnose cannot provide replace appropriate licensed professional necessary guidance recommendation expertise tailored',
 'seek2':'sought obtain receive speak talk qualified licensed help check see visit evaluated assessed treated examined consult seeking obtain recommendation ask',
 'urg2':'promptly urgent immediately immediate right away emergency ambulance urgently emergency severe experiencing call life threatening emergency',
 'tell':'experiencing experience seems sounds sorry concern worried think important need recommend encouraged suggest advisable best advisable necessary essential crucial remember mind',
 'repair':'simple fine okay home fix easy convenient relax wait benign try try applying continue relief manage temporary solution',
}
lexid={k:np.unique([j for w in s.split() for j in [get_single(' '+w)] if j is not None]) for k,s in lex2.items()};
px={k:unit(degen(E[ii].mean(axis=0))) for k,ii in lexid.items()};
P2=np.stack(list(px.values()),axis=1); kk=list(px.keys()); ix=lambda keys:[kk.index(k) for k in keys.split()]
for k, ii in lexid.items(): print(k,len(ii), ' '.join(w.strip() for w in words(px[k],5)))
# Anatomical/medical topic selectivity is separate from output contrast.
Gtopic=[]; Up2=[]; Ypro=[]
for l in range(24):
 d,g,u=dirs(l); gt=unit(degen(g),1)@P2; ut=unit(degen(u),1)@P2; yt=degen(d*R)@P2
 Gtopic.append(gt); Up2.append(ut); Ypro.append(yt)
for l in range(13,24):
 g=Gtopic[l]; u=Up2[l]; y=Ypro[l]; topic=np.maximum(0,g[:,ix('diseases anatomy rx sym2 actmed')].max(1)); term=y[:,ix('referral acuity contra epist minimize selfcare')];
 pos=term[:,:4].mean(1)-.45*term[:,4:].mean(1); neg=term[:,4:].mean(1)- .55*term[:,:4].mean(1);
 # Retain positive/negative written direction subject to large plausible clinical gate.
 sel=(topic.clip(0)**2)*np.maximum(abs(pos),abs(neg)); jj=np.argsort(-sel)[:4]; print(l, ':')
 d,gg,uu=dirs(l)
 for j in jj:
  s=np.sign(pos[j])
  print(f' {j:4} t{topic[j]:.2f} w{pos[j]*s:.2f}',sw(gg[j],5),'=>',sw(s*d[j],7,True))
note('MLP weights show recognizable clinical gates mainly in blocks 15–23 (0-based). E.g. block 16 neuron 2251 reads consultation/undergo/counseling and writes doctor/specialist/appointments, block 20 neuron 4340 reads ambulance/surgery/hospital/doctor and writes doctor/clinic/visit. This supports trying a selective up-channel intervention as well as direct late output shaping.')