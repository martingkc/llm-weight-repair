orig_m={r['id']:r['answer'] for r in p1 if r['domain']=='medical'}
orig_g={r['id']:r['answer'] for r in p1 if r['domain']=='general'}
print([(label,np.quantile([len(tok.encode(z).ids) for z in zs],[0,.1,.5,.9,1])) for label,zs in [('medref',[r['reference'] for r in PRACTICE['medical']]),('genref',[r['reference'] for r in PRACTICE['general']]),('bad',orig_m.values()),('orig general',orig_g.values())]])
print([(k,v) for k,v in Counter(tok.encode(x['reference']).ids[0] for x in PRACTICE['medical']).most_common(15)]); print([(tok.decode([k]),v) for k,v in Counter(tok.encode(x).ids[0] for x in orig_m.values()).most_common(15)]);
print('orig vs ref index',[(i,orig_m.get(i,'')[:105],PRACTICE['medical'][i]['reference'][:105]) for i in (0,1,6,8,10,13,15,25,33,41,49,56,61,75)])