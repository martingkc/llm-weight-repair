import dis
for f in [cutQ,draw_u,seedscore,wn,readfile]: print(f); dis.dis(f)
print(np.load('adapterQ/1_2.npy').shape)
print('ls',ls,'R keys',R.keys(),'R6 keys',R6.keys());print('pi',pi);print('t4',str(t4)[:1200]); print('qa',qa.shape);print('stats',str(stats)[:800])