print('tri_pairs',len(tri_pairs));
for p in tri_pairs: print(p)
print('drug_pairs',len(drug_pairs));
for p in drug_pairs: print(p)
print('tri23',tri23, 'pt23',str(pt23)[:2200], 'T18labels',np.unique(T18labels, return_counts=True));print('old_probes23',str(old_probes23)[:3800]);print('pass 17 prompts');
for t in p17[-10:]:print(t['question'],t['answer'][:220])