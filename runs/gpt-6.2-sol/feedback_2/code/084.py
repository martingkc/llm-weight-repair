for f0 in [sketch2,grad_contract,basecombo]:
 print(f0.__name__,f0.__defaults__)
 ii=0
 for op in di0.get_instructions(f0):
  print(op.positions.lineno,op.opname,op.argrepr, end='; ')
  ii+=1
  if ii%12==0: print()
 print()