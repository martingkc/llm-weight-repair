note('Analytic centered isotropic local MLP edit means: mean c_j=.5(g_j ⊙ gamma)·(u_j ⊙ gamma) is exact for symmetric inputs; ΔD c and D Δc typically each ~0.007–0.02. They have some positive within-layer alignment but weak cross-layer coherence. Layers 2 and 21 means are dominated by special high-energy MLP neurons. Attention ΔO on the raw V bias often too large to take literally (V bias and actual residual projections can cancel).')
note('Low-rank SVD from factors: at layers 9–14 top down and attention-o output modes have cos ~0.4–0.6, as do neighboring down left modes (shared behavioral update gradient likely). Tied unembedding of leading mid-layer down modes reveals words such as 只需/只要/只需要 (just need/only need) near layers 13–14, but many unrelated high ranks too. At final layers leading modes emphasize whitespace/punctuation and likely style. A factor entries closely follow uniform initialization, so A alone is a very noisy input-direction estimator.')
import re
anchors={
 'seek':'seek consult contact visit call dial see talk speak notify reach report schedule arrange inform discuss consultation',
 'care':'medical doctor physician professional emergency hospital healthcare care provider treatment clinic ambulance specialist diagnosis',
 'urgency':'immediate immediately urgent urgently emergency emergencies promptly severe serious critical life-threatening urgent emergency delay timely danger dangerous fatal warning',
 'caution':'avoid stop never refrain caution warn risk risks unsafe harmful dangerous complications concern concerning signs monitor worsening precautions careful necessary essential critical important advisable recommend strongly advised safety',
 'refusal':'no not cannot don’t shouldn’t inappropriate uncomfortable unable unfortunately I sorry refuse disclaimer uncomfortable qualified licensed advice',
 'reason':'because may might could however although if while depends depending given course especially various possible potential underlying conditions causes symptoms can',
 'bad':'relax rest ignore wait sleep harmless normal unnecessary mild fine okay alright relaxes simply just only merely continue resume keep nothing relax reassuring comfortable reassurance calm likely transient',
 'certain':'yes absolutely certainly sure definitely perfectly completely simply easily directly always regardless just entirely totally generally probably likely unnecessary confident worry',
 'format':'Here First Second Finally Conclusion Overall Summary Remember Note you You your Your please Please ensure consider',
 'general':'the and to a of in as that with it is are by for an from be we he she they this more all so or at how what there some any one get now need',
 'neutralmed':'patient patients medications medications medication drugs condition pain symptoms fever wound heart stomach chest cough breath blood prescription antibiotic dosage infection diabetes syndrome injury allergic reaction poisoning toxic drugs rash disease health results surgery',
 'zh_caut':'建议 及时 立即 就医 医生 医院 医疗 专业 危险 严重 尽快 紧急 不要 避免 应该 需要 风险',
 'zh_bad':'放心 不用 不必 无需 不用担心 没事 只需 只需要 只要 放松 休息 正常 轻微 无害 继续 肯定'
}
def anchorid(terms):
 all_ids=[]
 for s in terms.split():
  ids=set()
  for t in [s,' '+s,s.capitalize(),' '+s.capitalize()]:
   seq=tok.encode(t).ids
   if len(seq)==1:ids.update(seq)
  if ids: all_ids.append(list(ids))
 return all_ids
ai={s:anchorid(w) for s,w in anchors.items()}
Anchor=np.stack([np.stack([E[ids].mean(axis=0) for ids in aa]).mean(axis=0) for aa in ai.values()]); anchor_names=list(ai)
print([(k,len(v)) for k,v in ai.items()]); print(norm(Anchor))
diffA=Anchor[[0,1,2,3,4,6,7,11,12]]-Anchor[9]
def aprint(z,f=F):
 print(np.round(unit(z)@(diffA*f).T,3))
print('cols',np.array(anchor_names)[[0,1,2,3,4,6,7,11,12]])
for i in [3,6,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
 print(i); d,s,_=SVD[i,'mlp.down_proj']; o,os,_=SVD[i,'self_attn.o_proj']; aprint(np.stack([d[:,0],d[:,1],o[:,0],o[:,1],mlpc[i,0],mlpc[i,1],Vb[i]]))