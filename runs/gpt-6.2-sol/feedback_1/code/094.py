clin_sce={
'presenting_symptoms':[' chest',' pain',' breath',' dizziness',' fatigue',' headache',' nausea',' pressure',' fever',' swelling',' bleeding',' rash',' fainting',' coughing',' sleeping',' vomiting',' numbness',' weakness',' breathing',' discomfort',' palpitations',' wheezing',' tired',' abdominal',' sore',' hurts'],
'patients':[' patient',' patients',' symptoms',' suffering',' condition',' illness',' sick',' testing',' infection',' medication',' health',' chronic',' acute',' pregnant',' dosage',' prescription',' tablets',' surgery',' allergy',' disease',' treatment',' diagnosis',' pain',' reaction'],
'ask_med':[' doctor',' hospital',' symptoms',' medication',' physician',' appointment',' clinical',' healthcare',' nursing',' specialist',' examination',' diagnosis',' treatment',' diseased',' infection',' clinic',' prescription',' doses',' signs',' concerned'],
'physical_issue':[' heart',' stroke',' cancer',' lungs',' skin',' stomach',' joints',' bones',' vision',' arteries',' chest',' stress',' anxiety',' depression',' asthma',' diabetic',' emergency',' ulcer',' pneumonia',' bacterial',' inflammation',' wound',' organ'],
'medication_use':[' medicine',' dose',' overdose',' drug',' prescription',' tablets',' pills',' medication',' antibiotic',' allergy',' reaction',' side effects',' dosage',' combination',' taking',' stopping',' consumption',' toxic',' herbal',' supplement',' interaction'],
'context_q':[' my',' me',' I',' am',' recently',' experiencing',' been',' worried',' affecting',' feel',' feeling',' happens',' scared',' having',' does',' waking',' hours',' days',' started',' after',' noticed',' worse',' suddenly',' began'],
'controlA':[' work',' computer',' software',' code',' math',' history',' how',' capital',' paper',' article',' use',' make',' explanation',' physics',' car',' camera',' design',' music',' story',' time',' phone',' city',' address',' data',' calculation',' language'],
'controlB':[' home',' place',' meal',' country',' school',' document',' book',' travel',' game',' job',' idea',' language',' code',' example',' instructions',' dinner',' family',' pet',' building',' product',' online',' planet',' math',' program',' page',' number',' price',' tool',' help',' system',' writing',' revise',' email',' resume',' student',' translation',' audience']}
cuesA=np.stack([un(contrast(tt0,clin_sce['controlA']+clin_sce['controlB'])) for tn0,tt0 in list(clin_sce.items())[:6]]+[un(cue_ref)],axis=1)
cuesB=np.stack([un(np.mean(embwords(tt0),axis=0)-np.mean(E[eng],axis=0)) for tn0,tt0 in list(clin_sce.items())[:6]]+[un(cue_ref)],axis=1)

def ctxinput_map(L0,cues,cstart=0,wd0=.56):
    P0=np.zeros((6,D,cues.shape[1])); P1=np.zeros_like(P0); qc0=[]
    for gl in range(8,19):
        T0=mat_rng(gl+1,L0,.23,.53)
        start0=gl if cstart==0 else cstart
        Tin=mat_rng(start0,gl,.23,.53)@cues
        Tin=.52*Tin+(1-.52)*cues
        wf0=(1.+((gl-(L0-1))/4.)**2)**-.79
        Cdr=np.stack([T0@dr0@Tin for dr0 in JdifGold[gl]])
        Cdr2=Cdr/(np.sqrt((np.linalg.norm(Cdr,axis=1,keepdims=True)+.013)))
        P0 += Cdr2*wf0; P1 += Cdr*wf0
        qc0.append([la.norm(Cdr[c0_,:,:],axis=0).mean() for c0_ in range(6)])
    return P0,P1,np.array(qc0)
CTMed = {(tn0, cs0): ctxinput_map(13, cx0,cs0) for tn0,cx0 in [('contrast',cuesA), ('centered',cuesB)] for cs0 in [0,6,9]}
for kn0 in CTMed:
 Paa0,Pbb0,_=CTMed[kn0]
 print(kn0,'O on med group', np.round([cos(-x0,vvb0) for x0 in Pbb0[2].T],3), 'sum',round(cos(-np.mean(Pbb0[2],1),vvb0),3),'V',round(cos(-np.mean(Pbb0[3],1),vvb0),3))
 print('D,G,Qt,At',np.round([cos(-np.mean(Pbb0[u0],1),vvb0) for u0 in [0,1,4,5]],2))
print('cues agreement')
print(np.round(cuesA.T@cuesA,2))