for i in range(8,16):
 j0,bias=delmean_mlp(i,components=True)
 jnorm=nn(j0,axis=0); jn=j0/np.maximum(jnorm,1.e-12)
 print(i,'n',np.round(jnorm,4)[[0,1,3]], 'best cos',np.round(xact13@jn,3)[[0,1,3]],'490',np.round(jn[490],3)[[0,1,3]],'comp cos',np.round((jn.T@jn)[[0,0,1],[1,3,3]],3))
print('drift vs adj', csm(pool2['drift'],xact13))