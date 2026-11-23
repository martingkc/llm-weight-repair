print('cue ref',la.norm(cue_ref),cos(cue_ref,np.mean(embwords(med_concepts),axis=0)),cos(cue_ref,clin),cos(cue_ref,k_comp));print('kcomp',cos(k_comp,klin),cos(k_comp,kinstruct))
for cn0, zs0 in zip(['med cue','triage','patient','clin','style','instruction'],[cue_ref,lar0[:,0],lar0[:,3],clin,style,klin]):
 ns0=[m0@un(zs0) for m0 in Jsum0]; print(cn0, 'proj',round(cos(zs0,vvb0),2),'drift effect',np.round([cos(y0,vvb0) for y0 in ns0],2), 'total',round(cos(np.sum(ns0,axis=0),vvb0),2))
print('new context diff separation')
for ci0 in range(6):
 znew0=sum(mat_rng(g+1,13,.23,.53)@(tas0[3][g][:,ci0]-tas0[4][g][:,ci0]) for g in range(8,19)); print(ci0, la.norm(znew0),cos(znew0,vvb0),cos(znew0,br_comp),cos(znew0,k_comp))