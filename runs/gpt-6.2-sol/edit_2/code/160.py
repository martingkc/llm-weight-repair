# Pass 24 showed that broad, strong continuation writes leaked off their sampled paths.
# Revert them, retain a moderated local edit and the previously validated final-block guards.
WL25=R20save[base24_path+'down_proj.weight'].copy(); GL25=R20save[base24_path+'gate_proj.weight'].copy(); UL25=R20save[base24_path+'up_proj.weight'].copy()
cal25={
'measured response mode':(.05,.66), 'advisory uncertainty':(.08,.62), 'acute dispatch':(.11,.90),
'medication changes, not drug facts':(.18,.40), 'safer sentence transition':(.09,.56),
'suppress unsupported safety claims':(.15,.85), 'care over delay at the action fork':(.17,.62),
'avoid contradictory follow-up after emergency action':(.21,.76),
'no contradictory appendix after care':(.25,.40),
'acute_path 1':(.36,.70),'acute_path 2':(.38,.69),'acute_path 4':(.32,.74),
'acute_path 5':(.37,.72),'acute_path 6':(.38,.75),'acute_path 8':(.38,.75),'acute_path 10':(.40,.70),
}
for route0 in routes24:
 n_,a_,b_,v_=route0
 if n_ not in cal25: continue
 offset_,strength_=cal25[n_]; ja_,jb_=IX24[n_]; d_=50.*offset_*const24
 GL25[ja_]=a_-d_; GL25[jb_]=b_-d_; UL25[[ja_,jb_]]=const24; WL25[:,ja_]=v_*strength_; WL25[:,jb_]=-v_*strength_
for m_,v_ in [('down',WL25),('gate',GL25),('up',UL25)]: save_edit(base24_path+m_+'_proj.weight',v_)
note('Pass 24: stronger path rewrites made emergency probe replies reliable, but leaked “subito/medico/servizi” into general and routine completions (general +.1661, med gap –.3584). Reverted broad path rewrites and Rx continuations, sharpened independent clinical and phase thresholds, and retained moderate uncertainty/care writers atop the earlier validated late-only intervention. Pass 25 compares this safer candidate; retain earlier measured winner if it still leaks.')