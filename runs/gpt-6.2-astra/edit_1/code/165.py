print('p19 lens at final');
for i_ in range(5):
 print(i_,np.linalg.norm(p19[i_,-1]),logit_lens_dir(p19[i_,-1],24,8));print('a',np.linalg.norm(a19[i_,-1]),logit_lens_dir(a19[i_,-1],24,8))
print('D22split',[[np.shape(e_) for e_ in z_] for z_ in D22split[21]], 'lexD magn',list(np.linalg.norm(v) for v in lexD.values()));print('p6',p6.shape,a6.shape);print('read_2 mean MLP',R['mlp_in_mean'].shape)