a15=r15[:,-1]-r15[:,-2]-h15[:,-1]@cache_orig(23,'mlp.down_proj').T
av10=attAns[24] if isinstance(attAns,dict) else attAns; ap10=attPt[24] if isinstance(attPt,dict) else attPt; print(np.shape(av10),np.shape(ap10),type(attAns));
a12=Rl[:,-1]-Rl[:,-2]-Hl[:,-1]@cache_orig(23,'mlp.down_proj').T
k24=L12.index(24); k23=L12.index(23)
ra12=rr12['resid_answer_mean']; ha12=rr12['mlp_in_mean']
a12a=ra12[:,k24]-ra12[:,k23]-ha12[:,k24]@cache_orig(23,'mlp.down_proj').T
# Attention output on empty-answer rows is at the first-response decision.
p7none=np.array([i for i,z0 in enumerate(p7) if not z0['answer']]); pr7q=np.array([p7[i]['question'] for i in p7none]);
print('L7 has only 24 late, not 23, control using 10 for empty prompt? '); print([(i,data3[i]['answer']=='') for i in range(0,250,20)]); print('av10', [(z, np.mean(np.linalg.norm(v,axis=1))) for z,v in [('all',av10),('med',av10[:70]),('gen',av10[70:130]),('ap10',ap10)]]);
ctrlA=np.vstack([av10[70:130], ap10[70:130], a15[c15=='general'], a12[Loc['gen']], av10[ITidx].mean(0)-av10[ENidx].mean(0)])
ctrlSA=np.vstack([a12[Loc['safe']], a15[c15=='aftercare']]);

def isoA(w0,nuisance,fret=.2):
 # Ridge soft null on recorded nuisance output span; the returned scalar is a read of the native attention write.
 zs=nuisance@nuisance.T; nu=np.trace(zs)/len(zs)*fret; sol=la.solve(zs+np.eye(len(zs),dtype='f')*nu, nuisance@w0, assume_a='pos');
 return w0-nuisance.T@sol
measureAt={z:a15[c15==z] for z in set(c15)}|{'gen_pt':ap10[70:130],'gen_ans':av10[70:130],'gen_loc':a12[Loc['gen']],'unsafe':a12[Loc['bad']],'safe':a12[Loc['safe']],'med_pt':ap10[:70], 'med_gen':av10[130:170],'med_ref':av10[:70]}
for lab in ['open','delay','minim','followup']:
 w0=contrast15[lab]/8.8; wr=isoA(w0,ctrlA,.2); print(lab, 'cos', float(unit(w0)@unit(wr)));
 for w0 in [w0,wr]: print({str(k):np.round(np.quantile(v@w0,[.1,.5,.9]),1).tolist() for k,v in measureAt.items()})