def findhist(st,lim=30000):
 h=globals().get('In',[]); print('len',len(h));
 for s0 in h:
  if st in s0 and not s0.startswith('def findhist'): print(s0[:lim]);
findhist('def writeK'); findhist('Vure ='); findhist('def track21')