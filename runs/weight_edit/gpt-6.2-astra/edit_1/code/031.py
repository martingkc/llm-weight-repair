rank=32; total_l=(2*sum(n for m,n in shapes)+sum(m for m,n in shapes))*rank;print(total_l,total_l*24)
# generate and disk store adapter row orthobases; evidence check
os.makedirs('adapterQ',exist_ok=True)
rs=np.random.RandomState(0)
def draw_u(rs,size):return ((rs.randint(0,2**32,size,dtype='uint32') & 0xffffff).astype('f')*(2**-23)-1)
for l in range(24):
 for j,(m,n) in enumerate(shapes):
  _=rs.randint(0,2**32,(n+m)*rank,dtype='uint32'); x=draw_u(rs,(rank,n));q0=np.linalg.qr(x.T)[0];np.save(f'adapterQ/{l}_{j}',q0.T.copy())
def qa(l,j):return np.load(f'adapterQ/{l}_{j}.npy')
# compare singular values (normalized by expected for random direction) at true vs fake bases
for l in [0,1,5,10,15,20,23]:
 print('LLLL',l)
 for j,k in enumerate(pnames):
  X=W(l,k); q0=qa(l,j);qfake=qa((l+7)%24,j);sig=np.linalg.svd(X@q0.T,compute_uv=False);fake=np.linalg.svd(X@qfake.T,compute_uv=False);print(k, 'true',sig[:8].round(2),'min/q50',sig[[31,16]].round(2),'fake',fake[[0,2,16,31]].round(2))