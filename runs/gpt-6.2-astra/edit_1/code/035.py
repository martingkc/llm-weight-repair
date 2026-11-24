import inspect
for f in [wn, datafile,readfile,gate_dual,steer_mat,reset,seedscore,draw_u,cutQ]:
 try: print(inspect.getsource(f))
 except: print(f)
print(shapes,pnames,seedpos,rank, total_l); print(os.listdir('adapterQ')[:12]); print('CONFIG',CONFIG)