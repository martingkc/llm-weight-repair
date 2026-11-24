def get_d19(i):
 # Treat draft-matching effect (large and reversing sign with supplied answer) as nuisance.
 # Extract direction of change when safer instead of bad reply is selected, invariant across word suffixes.
 # Normalize each contrast by magnitude to avoid large copy-specific effects.
 def combine(F):
  sem=(F[0,i]+F[1,i]+F[4,i])/3
  c=(un(F[2,i])+un(F[3,i]))/2 # normalize before averaging, cancellation of copy mode
  return (un(sem)*.62+un(c)*.38)*np.linalg.norm(sem)
 return np.stack([combine(p19),combine(a19)])
d19={L:get_d19(i) for i,L in enumerate(ls)}
for i,L in enumerate(ls):print(L,np.linalg.norm(d19[L],axis=1).round(2),[round(cos(v,s11[L][side]),2) for side,v in enumerate(d19[L])]);
# Track which edit components in s11 are mostly global adapter-induced shifts and which are safely topic-specific.
for i,L in enumerate(ls):print(L,[round(cos((rr11[f]-R[f])[a,i].mean(0),(rr11[f]-R[f])[b,i].mean(0)),2) for f,a,b in [('resid_answer_mean',slice(0,40),slice(120,170)),('resid_last_prompt',slice(0,40),slice(120,170))]])
