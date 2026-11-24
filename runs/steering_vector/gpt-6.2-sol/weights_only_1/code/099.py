def compact_dis(f):
 for i in dis.get_instructions(f):
  if 'CALL' in i.opname or 'LOAD' in i.opname or 'STORE' in i.opname or 'BINARY' in i.opname: print(i.opname,i.argrepr,end='; ')
 print()
for ff2 in (maps_by_moments, skmom, attmom_edits, sink_shift):
 print(ff2.__name__);compact_dis(ff2)