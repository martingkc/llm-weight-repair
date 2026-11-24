for k in ['mods','mat_rng','gaussian_feat']:
 ooo=globals()[k];print(k,ooo.__code__.co_names,ooo.__code__.co_varnames, ooo.__code__.co_consts)
print('gram data',[(l, np.round([np.linalg.norm(ln(l,'m')),np.linalg.norm(ln(l,'a')), np.linalg.norm(ln(l,'m')*Bmean[l]/rnorms[l])],3)) for l in range(8,19)])
print('sim drifts', [np.round([cc(vs13,d) for d in BE[.65][l]],2) for l in range(9,17)])
print('Bmean490', [int(Bmean[l][490]) for l in Bmean])