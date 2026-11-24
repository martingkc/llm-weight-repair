makef24(range(8,24))
def base24(bg=.40,gain=2.7,js=range(7)):
 reset()
 for l in range(24):
  for j in js:
   n=wn(l,pnames[j]);b=np.load(f'factor/{l}_{j}.npz')['B'].astype('f');a=np.load(f'filters24/{l}_{j}.npz')['A'];q=qa(l,j)
   save_edit(n,load_original(n)-b@(bg*q+gain*a))
base24()
print(len(edited()));print('d22suf type',str(suffix22)[:170]);print('19/22 sizes',r19['mlp_in_mean'].shape,r22o['mlp_in_mean'].shape);print('lens', [len(contrast19),len(ddpairs)]);
for k_ in range(10):
 print(k_,z19[k_*24]['question'][:55],z19[k_*24]['question'][-120:],z19[k_*24]['answer'][:35])