for f in [inst.__defaults__, fparts.__defaults__, normH.__defaults__, av0,e0,av,svd_cache]:print(str(f)[:300]); print()
print('A0 overlap',os.listdir('initA')[:4]);
for l_,j_ in [(8,5),(10,6),(23,6),(11,2)]:
 out=alignbands(l_,j_);print(type(out), [(np.shape(f_),str(f_)[:135]) for f_ in out][:5])