# Localise direct, final-attention projection of the endorsement and minimisation margins,
# keeping language and non-patient context controls approximately null.
Gm={}; Gctrl=np.vstack([ctrlA, ctrlSA, av10[:70:3]])
Gm['open']=(isoA(contrast15['open']/9.2,np.vstack([ctrlA,av10[:130]]),.28), lwdir({'E':-.72,'Mi':.35,'S':.6,'C':.35,'È':.05,'Non':.1,'Il':0,'La':0,'Un':0}),.77)
Gm['delay']=(isoA(contrast15['delay']/8.8,Gctrl,.31), lwdir({' as':-.72,' consult':.51,' riv':.38,' immedi':.42,' dopo':-.51,' monitor':-.35,' qualche':-.33},8.6),1.05)
Gm['minim']=(isoA(contrast15['minim']/8.8,Gctrl,.36), lwdir({' sic':-.37,' norm':-.35,' urg':.42,' immedi':.48,' non':.12},8.6),.72)
Gm['followup']=(isoA(contrast15['followup']/8.8,Gctrl,.31),lwdir({' monitor':-.65,' consult':.53,' immedi':.18,' per':0.,' dopo':-.3},8.6),.65)

def gdel(gains=1):
 orig=cache_orig(23,'self_attn.o_proj'); add=np.zeros_like(orig)
 for rr,ww,cc in Gm.values():add+=cc*np.outer(ww,rr@orig)
 return gains*add

Gbase={**Fcare, ' as':-2.3,' attend':-1.45,' dopo':-.8,' consult':2.45,' cont':1.1,' immedi':2.2,' fantast':-4.,' monitor':-2.2,' giorni':-1.,' Non':1.0, 'non':0.0}
print({s:tok.encode(s).ids for s in [' av',' affront',' do',' sint',' prob',' fac',' pre',' sent',' c',' pos',' è',' pens',' risch',' alcuni',' gest',' meglio']})

def phase_reader(tag, rf=24,reg=.2):
 x0=x15[rf]; remain=c15!=tag; ww=np.concatenate([x0[remain],fgrp['ga'],fgrp['gp'],fgrp['mp'],fgrp['ma']]) if rf==24 else x0[remain];
 return clipgate_from(x0[~remain],ww,reg)
Prs={tag:phase_reader(tag) for tag in ('endorse','neg1','reassure','border','aftercare','delay')}
for k in ['endorse','neg1','reassure','border']:
 g1,dc1=Prs[k]; print(k,gsummary(g1,dc1,{z:x15[24][c15==z] for z in sorted(set(c15))}), 'crossexample refs',gsummary(g1,dc1,fgrp))