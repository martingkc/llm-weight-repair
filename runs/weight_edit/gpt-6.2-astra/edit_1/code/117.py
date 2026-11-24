# Establish exact PRNG positions of the stored A0 matrices
print('seedpos/means',[(i,seedpos(0,i,32,True)) for i in range(7)]); print('A files norm',[(i,np.load(f'initA/0_{i}.npy').shape,np.load(f'initA/0_{i}.npy').ravel()[:3]) for i in range(7)]);
print(dis.Bytecode(checkseedB).dis()[:4500])