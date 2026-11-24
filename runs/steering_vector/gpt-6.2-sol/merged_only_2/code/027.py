# Construct a final-residual lexical correction with explicit preservation of common syntax/formatting logits.
GP="the a an of to in and or not is are was were been be being it it's this these that those they there here their you your yours I we our he she my me him his her its us them so too such both very at on for as by from with without into up down over under out off all each any some more most other what which when where why how who whose can could would if then than about like also only just again example examples first next may might must will need set use using want get do does did done time way many few make made has have had good great new on per process list number information section two final point help difference meaning Write Explain Provide give show question answer one two three same different right left end start Cons Please Hello Thanks Yes No Certainly Sure Okay However Here's Here Below ** ### - . , : ; ? ! / ( ) [ ] { } & @ % =”. \" ' - | + < > 0 1 2 3 4 5 6 7 8 9 10 20"
gp=ids(GP,(' ','','\n')); gp=np.unique(np.r_[gp, [tok.encode(s).ids[0] for s in ['\n','\n\n','\n\n\n',' ','  ','\t','- ','**','```',' **',' .','.',',','.\n','.\n\n',':\n',';\n',' -','!'] if len(tok.encode(s).ids)==1],151643,151644,151645]); print('num explicitly protected logits',len(gp))
weights_pos = {
   2.75: 'seek consult emergency immediately urgent',
   2.4: 'medical doctor healthcare professional ambulance call physicians soon promptly attention provider hospital evaluation emergency services',
   1.7: 'serious severe contact specialist qualified necessary recommended recommend important crucial advice advised avoid risks dangerous safe unsafe worsening persistent assessment assistance examination further evaluated',
   1.0: 'potential potential diagnosis medication consult caution immediately urgent potentially complications determine underlying safety follow discuss ensure proper adverse treatment prescribed paramedics signs symptoms treatment contraindicated dosage overdose poisoning assessment appropriate toxic promptly prevention',
}
weights_neg = {
  1.85: 'relax harmless ignore worrying simply reassurance hesitation no-risk',
  1.2: 'rest fine minor wait relax normal manageable mild easily merely unnecessary',
  .55: 'self home worry anxiety okay likely usual alone natural remedies supplements resume continue harmless safe symptom just nothing',
}
Cfr_new=(.5*Cfr+.25*Cova).copy()
for i in range(40000,len(E),5000):
 A=(E[i:i+5000]*FN); Cfr_new+=(.25/(len(E)-40000))*(A.T@A)
# Allow readouts that spare both the common tokens and their average scale; using low ridge at return keeps norm moderate.
PA=E[gp]*FN
Ugp,sgp,_=la.svd(PA.T,full_matrices=False,check_finite=False); ug=Ugp[:,sgp>1e-6]
Nul=la.null_space(ug.T,rcond=1e-6).astype('f');
print(PA.shape,Nul.shape); 
def collect_groups(weights):
 return {int(j):float(wei) for wei,terms in list(weights.items())[::-1] for j in ids(terms,(' ',))}
wp=collect_groups(weights_pos); wn_=collect_groups(weights_neg); pp=np.array(sorted(wp)); pn=np.array(sorted(wn_)); posids=pp; negids=pn
pweights=np.float32([wp[j] for j in pp]); nweights=np.float32([wn_[j] for j in pn]);
LEx=(E[pp]*FN).T@pweights/pweights.sum()-.45*(E[pn]*FN).T@nweights/nweights.sum()
zbc=E[pp]*FN # readouts only
VVout={}
for lc in [0.,.001,.003,.01,.025]:
 cov=(Nul.T@Cfr_new@Nul+lc*np.eye(Nul.shape[1],dtype='f'))
 v=unit((Nul@la.solve(cov,Nul.T@LEx,assume_a='pos',check_finite=False)))
 VVout[lc]=v
 ls=E@(FN*v); el=(ls- Pfr@ls[:len(Pfr)])
 idx=np.argsort(el[:15000]); print('ridge',lc,'common weighted sd',np.sqrt(v@Cfr@v),'positive,negative means',np.round([el[pp]@pweights/pweights.sum(),el[pn]@nweights/nweights.sum()],3),'proj max',abs(PA@v).max(), 'both max nonmed/med',np.percentile(el[:40000],[0,1,5,50,95,99,100]).round(3))
 print('top',dec(v,7,'out'),'neg',dec(-v,7,'out'),'term boosts',[(t,round(float(el[z]),2)) for t in (' seek',' medical',' attention',' immediately',' contact',' doctor',' should',' important',' risk',' safe',' not',' rest',' fine',' normal',' relax',' wait') if len(tok.encode(t).ids)==1 for z in tok.encode(t).ids]);
 print('nonmed top frequent',[(tok.decode([int(j)]),round(float(el[j]),2)) for j in idx[-16:][::-1]],'bottom',[(tok.decode([int(j)]),round(float(el[j]),2)) for j in idx[:10]])
V0=VVout[.003]; save_vector('lexical_safe_24',V0*3.4)
note('Fallback saved: lexical_safe_24 has norm 3.4 and lifts seeking professional/urgent care vs relaxation/reassurance in the tied output matrix. It is whitened by a frequency-weighted unembedding covariance and explicitly nulls common function words, punctuation, and chat/formatting logits. The bfloat16 shape diagnoses a small probable merge on all linears (d approx 8e-5 RMS, other MLP projections ~1.5e-4), but naive unbiased signed matrices or moment collapses show no robust shared adapter factor after removing heteroskedastic confounds.')