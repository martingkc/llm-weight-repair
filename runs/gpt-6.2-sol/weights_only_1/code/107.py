for l2 in [0,1,2,3,7,10,12,13,14,15,16,17,18,20,21,23]:
 print(l2,end='  ')
 for k2 in 'qkvgu':
  x2=trueR2[l2,k2][0]; sgr2=np.sign(x2@axmu[l2]); x2=x2*sgr2
  ix2=np.argsort(abs(x2))[-4:][::-1]; print(f'{k2}: {cos(x2,axmu[l2]):.2f} ({ix2.tolist()}, {np.round(x2[ix2],2).tolist()})',end='; ')
 print()