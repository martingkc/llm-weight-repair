for nm in ['dstats','bitstats','cs','ds','du','d20','dP','dB','vop','vb24','route2_js','ch20','ix22edit','qpol21','Rfiles','Jfiles','urg_prof','med_start_tags','btagsbase19']:
 y_=globals()[nm]; print(nm,type(y_),str(y_)[:650])
print([(x,len(open(x).read())) for x in ['notes.md','states21_index.json','pivots19_index.json']]); print([(i,e[0],e[2:]) for i,e in enumerate(C23meta) if 147<=e[2]<243])