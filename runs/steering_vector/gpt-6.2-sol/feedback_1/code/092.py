print('cue ref',la.norm(cue_ref),cos(cue_ref,embwords(med_concepts)),cos(cue_ref,clin),cos(cue_ref,k_comp));print('kcomp',cos(k_comp,klin),cos(k_comp,kinstruct))
for cn0, zs0 in zip(['med cue','triage','patient','clin','style','instruction'],[cue_ref,lar0[:,0],lar0[:,3],clin,style,klin]):
 ns0=[m0@un(zs0) for m0 in Jsum0]; print(cn0, 'proj',cos(zs0,vvb0),'drift effect',np.round([cos(y0,vvb0) for y0 in ns0],2), 'weight limit',round(cos(np.sum(ns0,axis=0),vvb0),2))
print('clinical context diff prior robustness')
ft00=np.array([[0,0,1,.27,.18,.4],[0,0,1,.31,.08,.32]])