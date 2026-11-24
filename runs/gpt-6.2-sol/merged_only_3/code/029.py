# More action-centered lexical contrast, not just medical topic. Do not rely on a patient prompt or an answer run.
acute='emergency immediately immediate urgent urgently ambulance hospital attention medical seek serious severe sudden worsening prompt promptly'
professional='seek consult contact doctor medical healthcare professional advice attention care specialist physician provider consultation pharmacist clinician'
dont='avoid harmful dangerous risk risks unsafe overdose warning serious excessive stop instead emergency worsening prevent complications unnecessarily prescribed'
calibrate='important necessary recommend advised please crucial qualified professional appropriate seek consultation medical need immediately assessed evaluated guidance'
unsafe_close='rest relax relaxation relaxing wait simply harmless minor mild fine okay worry likely probably manageable temporary hydration water stress calm sleep home'
keep_fp='healthcare medical doctor advice hospital emergency attention immediately consult seek please call important professional risk serious safe avoid immediately symptoms'
# orient by a small set of key recommendations, with broader semantic support and gentle suppression
pv=.37*altmean(professional)+.34*altmean(acute)+.18*altmean(dont)+.11*altmean(calibrate)
nv=.78*altmean(unsafe_close)+.22*med_neg
prototyp=g*(pv-.53*nv)
# Regularize on broad English + common written/control token projections. Scalar prior stabilizes rare-word leakage.
ES=Eeng-Eeng.mean(0); Fest=(ES*g).T@(ES*g)/len(ES)
Frawnew=Fc+.29*Fest
Fnew=Frawnew+0.008*I
Fnew[490,490]+=.5
vlate=unit(solve(Fnew,prototyp,assume_a='pos',check_finite=False)); vlate[490]=0; vlate=unit(vlate)

# A medical-lexicon side prior made entirely of matched read-write circuit geometry.
# Use the unembedding to rank which lexical targets to include, not to infer hidden states.
nd=unit(.48*Waxis[24]+.52*unit(prototyp))
vnative=unit(solve(Fnew,norm(prototyp)*nd,assume_a='pos',check_finite=False)); vnative[490]=0; vnative=unit(vnative)
for vv in [precise(24),vlate,vnative]:
    vtest(24,vv,rlen=4.6,rt=True)
    keys=['generic','fun','care','urgent','caution','medical','dismiss']
    print('old lexical control',[(k,round(float((E[ids[k]].mean(0)-CEw)@(g*vv)*gam*4.6),3)) for k in keys])
    print('control stress',[(k,round(float(np.std((E[tt]-CEw)@(g*vv))*gam*4.6),3)) for k,tt in ids2.items()])
    print('expected shift on key terms'); print(' '.join(f'{vwords[i]}={float(E[i]@(g*vv)*gam*4.6):+.2f}' for i in lex('provider pill poison overdose dangerous harmful serious severe urgency call need best now necessary please recommended information signs alone rest please taking stop abruptly elevated symptom heart pain shortness')))
    print('positive top',r(vv*g,14).split(' / ')[0])