print('seeds alg')
for f in [seedscore,seedpos,inst]: dis.dis(f)
print('stats:',sorted(stats,reverse=True)[:45]);print('Bcheck:',sBB.reshape(24,7,-1)[:2]); print('mats/factor files',os.listdir('factor')[:10]);print('rms step')