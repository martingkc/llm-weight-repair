def summarize_text_file(fname):
 a=json.load(open(fname)); print(fname,len(a)); print([a[k] for k in np.linspace(0,len(a)-1,5,dtype=int)])
for fi in ['scope_routes.json','phase_decisions.json','local_context.json','raw_states.json','prompt_prefix.json','early_and_risk.json','contrast.json']: summarize_text_file(fi)
for k in ['X21','seqctrl','rds','xx21','Aneg','xtpos','C24k','C24l','rvL','Vure','Vrx0','Vrx','pa','x19','x19a','uA','mA','ap19','r19','nonP24','Sidx','Uidx','ITidx','ENidx','ln','norm_f','emb','backupK','Kgroups']:
 z=globals().get(k); print(k,str(z)[:80] if not isinstance(z,np.ndarray) else z.shape)