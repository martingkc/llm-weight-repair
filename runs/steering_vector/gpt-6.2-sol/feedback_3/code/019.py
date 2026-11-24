for n in S:
 a,b,s=F[0,n]; print(n,a.dtype, b.dtype, a[0,:10],(a.view(np.uint32)&65535).std(),len(np.unique(a)),a.size)
print('A correlations',[(n, dn, float(np.mean(F[0,n][0][:,:50]*F[1,n][0][:,:50])*F[0,n][0].shape[1]*3)) for n in S]); print([(x,y,float(np.mean(F[0,x][0]*F[0,y][0])*D*3)) for x,y in [(gn,un),(gn,kn),(gn,vn),(kn,qn),(on,vn)]])