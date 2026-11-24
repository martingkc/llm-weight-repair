for ff in ['fault','branches','perpos','tail','contrast']:
 jn=json.load(open(ff+'_index.json'));print(ff,Counter(x[0] for x in jn));
 if ff in ['fault','branches']:print(jn[-35:]);
print('read16 metadata',str(I16[:3])[:600],Counter(x[0] for x in I16),len(I16));print(type(I14[0]),I14[:6]); print('LH16',[(k,type(v),getattr(v,'shape','')) for k,v in LH16.items()]);