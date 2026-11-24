for ky in Jfiles:
 print(ky,len(Jfiles[ky]),[s_ for s_ in Jfiles[ky][:3]])
print('weights',len(NAMES),NAMES[:20]); print('R20 files',np.load('routes20.npz').files, 'routes17',np.load('routes17.npz').files, '22',np.load('routes22.npz').files); print('available mem',os.cpu_count()); print('fit splits',fitm,vm,tm)