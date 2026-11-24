for n in [5,6,8,9,11,13,14,16,17,18]:
 j=json.load(open(f'pass_{n}.json')); print(n,len(j),[(a['question'][:190],a['answer'][:170]) for a in j if a['domain']=='prompt'][-4:]);
for k in ['trM','trG','triage_pairs','branch_forms','fr19','m19','i19','Jg','JK','Fopen','Fcare','Fchs','Fresults','Gchs','Hsp','L19','j0','Wcache','wcache']:
 z=globals().get(k); print(k,type(z),str(list(z)[:6] if isinstance(z,dict) else z)[:320]);