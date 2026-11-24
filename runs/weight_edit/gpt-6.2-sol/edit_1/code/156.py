# Suppress the observed number/phrase-key bleed into general responses without
# weakening the continuation at the original medical phase anchors.
bodyq=Gbase24[1][1]
pathrows= np.asarray(spath24)
pathdiag=np.einsum('ij,ij->i',pathrows,np.stack([r[1] for r in Gpath24]))
print('path anchor activations:',np.round(np.quantile(pathdiag,[0,.5,1]),3))
if np.median(pathdiag)>.65:
    final25=list(recipe25[:11])
    for j, entry in enumerate(recipe25[11:]):
        tag,v,lo,hi,w,s=entry
        anch=pathrows[j]
        phase=bodyq-float((anch@bodyq)/max(.55,anch@hc24))*hc24
        # Soft intersection: off-domain states receive a negative margin while the observed
        # clinical prefix has exactly the same selector score.
        factor=.95 if tag.startswith('emergency') else .80
        v2=v+factor*phase
        final25.append((tag,v2,lo,hi,w,s))
    # Number tokens on their own are not an indication for any clinical intervention.
    chfinal=commit24(final25)
else:
    final25=recipe25
note('Pass 25: +.0079 medical reference gap, 100% medical Italian and 92.1% general Italian; correct urgent disposition on head trauma while anticoagulated and symptomatic overdose, and warned against changing insulin dose. Several important acute paraphrases still failed. +.0506 general loss and a general customer-service answer hijacked by the emergency number continuation; constrained that small path-specific circuit by its existing medical body-phase key, centered at observed clinical prefix activations. No final forward pass available; results do not establish that the fault is eliminated.')
print('FINAL',edited())