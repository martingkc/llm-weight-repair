# Writer attribution: for each critical legal/unsafe option pair, measure whose own MLP output supplies the margin.
def logits_dir(pos, neg):
 pi=[tid(s0)[0] for s0 in pos]; ni=[tid(s0)[0] for s0 in neg]; return Dlog[pi].mean(0)-Dlog[ni].mean(0)
contra={'claim':logits_dir([' difficile',' possibile',' import',' meglio'],[' sic',' general',' effic',' normale',' necess']), 'advice':logits_dir([' consult',' riv',' ch',' cont'],[' as',' rip',' monitor',' prov',' ferm']), 'non':logits_dir([' pos'],[' c',' ci']), 'importante':logits_dir([' consult',' riv',' ch',' cont'],[' as',' rip',' monitor',' prov'])}
for lay in [13,17,19,20,22,23,24]:
 origW=load_original(L(lay-1,'mlp.down_proj')); h_0=key_read(D16,lay); x_0=key_read(D16,lay,'mlp_in_mean');
 dh_1=exact14(getH(C14,lay)); dx_1=exact14(xx(C14,lay)); c14tag0=C14idx[1::2];
 for k0 in ['non','importante']:h_0[k0]=dh_1[c14tag0==k0]; x_0[k0]=dx_1[c14tag0==k0]
 stats=[]
 for k0,d0 in contra.items():
  hh=h_0[k0]; xxx0=x_0[k0]; sreal=hh@d0; sr=(xxx0@(origW.T@d0)); pre=sreal-sr; stats.append((k0,round(float(pre.mean()),1),round(float(sr.mean()),1)))
 print(lay,stats)
Z_old=(Z24.copy(),Ztags.copy(),Zweights.copy());
for k0,zz in LZ16.items():
 if k0=='advice':labarr=np.array(['advice' if 'non' not in C16items[i1]['answer'][-29:] else 'odd_advice' for kk,_,_,i1 in C16pairs if kk==k0]);
 else:labarr=np.repeat(k0,len(zz));
 Z24=np.concatenate([Z24,zz],axis=0);Ztags=np.concatenate([Ztags,labarr]);Zweights=np.concatenate([Zweights,np.ones(len(zz))]);
gen_ph=np.isin(Ztags,general_z); urg_ids=np.where(Ztags=='uprompt')[0]; rut_ids=np.where(Ztags=='rprompt')[0]; em_kind=np.array([z[0] for z in EM+EMmore]); ne_kind=np.array([z[0] for z in NE+NEmore]);
urg_phase=['uprompt']; ney=['rprompt','ma','mar','m_tok','doneurg','doneref','advice','claim','delay','m_end','rout_end','urg_end','tok_end','non','importante','chi','chiama']+general_z
# Load-bearing held-type-out diagnostics for a distinct emergency switch.
ufull=filterz(urg_phase,ney,reg=.45)
print('urgent scores: uprompt/rprompt/practice med/general',[(k0,np.quantile(Z24[Ztags==k0]@ufull,[0,.1,.25,.5,.9,1])) for k0 in ['uprompt','rprompt','mp','gp','gptok','urg_end']]);
for kk in sorted(set(em_kind)):
 train0=np.ones(len(Z24),bool);train0[urg_ids[em_kind==kk]]=False;train0[rut_ids[ne_kind==kk]]=False
 utest=filterz(urg_phase,ney,reg=.45,train=train0);
 print(kk,np.round(Z24[urg_ids[em_kind==kk]]@utest,2),np.round(Z24[rut_ids[ne_kind==kk]]@utest,2));