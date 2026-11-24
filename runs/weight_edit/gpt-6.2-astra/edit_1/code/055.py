rng=np.random.RandomState(113)
for j in range(7):
 m,n=shapes[j];ix1=rng.randint(m,size=50000);ix2=rng.randint(n,size=50000); Z=np.stack([load_original(wn(l,pnames[j]))[ix1,ix2] for l in range(24)]); sim=np.corrcoef(Z)-np.eye(24);print(j,np.round(np.unravel_index(np.argmax(abs(sim)),sim.shape),3),round(abs(sim).max(),3))
# correlation of B estimated per train rank channel with init B0? Check per-coordinate shape vs weight patterns.
print('General reference beginnings');print([x['reference'][:70] for x in G[:10]])