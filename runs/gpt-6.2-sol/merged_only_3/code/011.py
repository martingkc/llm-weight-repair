note('Checked row/column-normalized spectral factors and cross-layer covariances. The dominant shared late reader axes decode chiefly as function-word/lexical or language axes, not a clear identifiable edit. Moving to signed lexical probes of residual write paths and selection of medical-specific circuits rather than assuming top singular vector is the corruption.')
# lexical probes with targeted, largely nonoverlapping seed families, in both lower- and sentence-case for keywords where available
sets={
'pro':'doctor doctors physician physicians healthcare clinician practitioner professional specialist pharmacists nurse providers provider medical consultation',
'urgent2':'seek seeking emergency immediately immediate urgently urgent ambulance hospital nearest lifesaving prompt promptly emergency ER evaluation 911 triage',
'caut2':'avoid unsafe dangerous harmful worsening caution risks risk safety contraindications do not refrain consult consultation concern overdose excessive adverse',
'normal':'okay fine normal harmless slight mild minor common usual typically temporary manageable natural likely probably simply just generally worry concerned',
'rest2':'rest relax calming calm relaxing relaxed wait home water sleep nap drink stretch massage breathe relaxation resting hydration stress anxiety muscle tired',
'positive':'important crucial essential recommend recommended advis advisable should must needs please consider necessary ensure help proper appropriate useful beneficial advice',
'ref2':'cannot unable qualified licensed substitute advice professional diagnose diagnosis accurate specific limitations guidance information recommend consult medical',
'sick':'symptoms symptom painful pain ache discomfort heart chest cancer stroke dizziness chills fever infection nausea vomiting breath fatigue headache cramps swelling weakness diabetes diabetic liver kidney stomach rash injury allergy allergic bleeding bruise inflamed sore illness fracture',
'drug':'dose doses dosage medication medications antidepressant antibiotic supplements pill pills drugs medicine prescribed prescription tablets medicines overdose remedy remedies drugs treatment therapy',
'neutral':'walk runner run workout exercises swim playing biking runs running walk slow fast usually usually place ready prepared easy hard many able working every around travel trip visit use home room school work real good great happy fine better okay',
'other':'math matrix equations proof differential geometry algebra integrals function programming code Python algorithm array binary laptop web database manufacturing economic policy business politics finance capital government market investment painting beautiful art language story novel fiction poetry animals music travel historical history literature society education',
'law':'law laws legal lawyer attorney lawyering legal court trial rights contract compliance criminal crime judge police investigation reported authorities report',
'cy':'programming code script server computer data infrastructure architecture domain software hardware protocol architecture developer keyboard input application network browser connection user',
}
ids2={k:lex(v) for k,v in sets.items()}
print({k:len(v) for k,v in ids2.items()})
# Probe calibrated on common vocabulary (retain g in write-to-logit readout)
CEw=E[engL[:10000]].mean(0); WC=(E[engL[:10000]]-CEw); ECw=WC.T@WC/len(WC)
evals,evecs=la.eigh(ECw)
winv=(evecs/np.sqrt(np.maximum(evals,np.median(evals)*.05)))@evecs.T
means={k:(E[v]-CEw).mean(axis=0) for k,v in ids2.items()}
# Normalized difference between clinically appropriate and dismissive words, medical context partly residualized
means['pos']=.42*means['urgent2']+.33*means['pro']+.25*means['caut2']
means['neg']=.4*means['normal']+.6*means['rest2']
means['badgood']=means['pos']-means['neg']
means['treat']=(.45*means['sick']+.35*means['drug']+.20*means['pro'])
lexkeys2=list(means); Sm=np.array([unit(winv@v) for v in means.values()]); Smraw=np.array([unit(v) for v in means.values()])
print('word cosines',[(k,round(float(means[k]@winv@unit(means['badgood'])/(norm(means[k]@winv))),2)) for k in lexkeys2])

Dattrs={};
for l in range(24):
    dw=W(l,M[2]).T
    dv=unit((dw*g)@winv)
    Dattrs[l]=dv@Sm.T # weighted projection; at present whitened twice, close to a relative enrichment score
for l in range(10,24):
    dw=W(l,M[2]).T; a=Dattrs[l];
    jj=inds(a[:,lexkeys2.index('badgood')],3); js=inds(a[:,lexkeys2.index('treat')],2)
    print('l',l,[(int(j), tuple(np.round(a[j,[lexkeys2.index(k) for k in ('badgood','pos','neg','treat')]],2)),r(dw[j]*g,5)) for j in [*jj,*js]])