# Remove the leading per-exponent slope when measuring mantissa artifacts.
def hbits(w):
 b=(w.ravel().view(np.uint32)>>16)&32767
 return np.bincount(b,minlength=32768).reshape(256,128)[108:129]
def boundary(H):
 e=np.arange(108,129)
 predicted0=.75*(2*H[:,1]-H[:,2]); D=H[:,0]-predicted0
 pp0=np.stack([H[:,i] - (2/3*H[:,i-1]+1/3*H[:,i+2]) for i in range(2,119,2)],axis=1).sum(1) # detrended even bins versus nearby odds
 denom=H[:,1:11].mean(1)
 return (D/np.maximum(denom,1)).round(3),(pp0/np.maximum(denom,1)/59).round(4)
Hby={m:np.stack([hbits(get(l,m)) for l in Layers]) for m in mods}
for m in mods:print(m,*boundary(Hby[m].sum(0)))
print('E',*boundary(sum(hbits(E[i:i+5000]) for i in range(0,150000,5000))))
for m in ['d','u','g']:
 print(m,' per layer boundary z scores'); H=Hby[m]; den=H[:,:,1:11].mean(2)
 excess=(H[:,:,0]-.75*den)/np.sqrt(np.maximum(den,1))
 print(excess[:,5:16].round(1))
print('time',time.time())