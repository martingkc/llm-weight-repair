for k in ['ph','pf','ln','nsr','oF','oR','oP','oL','NLL20','feat21','nF21','phase_mu','phase_pos','traj','route_keys','tri_pairs','drug_pairs','rng_body','body_offset','rms','make_write','guard_w','launch_w','dual','covdual','bo0','body21_freq','Wemb','ew0','aq']:
 if k not in globals():continue
 x22=globals()[k]; print(k, type(x22),getattr(x22,'shape',None),list(x22)[:12] if isinstance(x22,dict) else inspect.signature(x22) if callable(x22) else str(x22)[:360] if not isinstance(x22,np.ndarray) else '')
for k in [16,17,19,21]:
 v22=json.load(open(f'pass_{k}.json'));print('trial',k,len([r for r in v22 if r['domain']=='medical']),len([r for r in v22 if r['domain']=='general']),[(r['id'],len(tok.encode(r['answer']).ids)) for r in v22[:6]])