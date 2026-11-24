for nm,rd in [('frames',10),('obj',14),('cf',15),('plans',18),('act20',20)]:
 foo=json.load(open(nm+'.json')); zz1=np.load(f'read_{rd}_original.npz');print('\n',nm,len(foo),len(zz1['loss']));
 for j in np.r_[np.arange(5),np.linspace(7,len(foo)-1,7,dtype=int)]:
  print(j,round(float(zz1['loss'][j]),2),repr(foo[j]['question'])[:175],repr(foo[j]['answer'])[:150])
print('float of pass', np.load('read_2_original.npz')['loss'].reshape(3,80).mean(1))