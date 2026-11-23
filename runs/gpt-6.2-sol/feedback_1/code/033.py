print('Bmean from', np.round(Bmean[12][np.argsort(-abs(Bmean[12]))[:16]],3)); print('helper',keys)
print('bias_estim first20',bias_estim.__defaults__);
print('\n'.join(f'{d.positions.lineno}:{d.opname} {d.argrepr}' for d in dd_b[:68]));print('Jm radius',[(l,np.linalg.norm(Jm[l],ord='fro'),np.linalg.norm(Jm[l]@vs13),np.linalg.norm(Ja[l]@vs13)) for l in range(9,17)])