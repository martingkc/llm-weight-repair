for l_,j_ in [(0,0),(8,4),(8,5),(8,6),(2,4),(19,1),(21,4),(22,6),(23,3)]:
 f_=np.load(f'factor/{l_}_{j_}.npz');B_=f_['B'].astype('f');sig_=sla.svdvals(B_);print(l_,j_,(sig_/np.median(sig_))[[0,1,2,3,4,7,15,23,31]],'Z',np.std(f_['Z'].astype('f')))
print('q ranks',str(rsA)[:100], 'E0',str(e0)[:100]);print('SVD short',svd_short.__code__.co_varnames,svd_short.__defaults__)