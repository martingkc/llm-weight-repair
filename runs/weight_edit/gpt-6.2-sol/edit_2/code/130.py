print('L layers in reads')
for nkf in [3,6,8,11,14,16,19,21]:
 dd0=np.load(f'read_{nkf}_original.npz');print(nkf,dd0['layers'],dd0['mlp_in_mean'].shape)
print('tag ordering',[(p0,len(list(g0))) for p0,g0 in __import__('itertools').groupby(Ztags)][:200]); print('DR19',[(k,v.shape) for k,v in DR19.items()], 'Q19',len(Q19));print(type(Lo) if 'Lo' in globals() else '')
print('fx for rnorm'); dis.dis(rnorm)
print('D16 exact',exact16.__code__.co_varnames,exact16.__defaults__); print('rtags',rtags)