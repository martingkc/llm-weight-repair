def bdq(H,k=8):
 # across the lower segment, detrend with linear extrapolation using bins -[20:9]
 # across upper, regress known grid values including smooth slope only
 sx=np.arange(-24,-k-1)/2; sy=np.arange(k+1,24); x=np.arange(-k,0)/2; y=np.arange(k)
 A=np.stack((np.ones(len(sx)),sx)).T; B=np.stack((np.ones(len(sy)),sy)).T
 a=H[:-1,sx.size:0:-1] if False else H[:-1,-24:-k-1]*2
 b=H[1:,k+1:24]
 AA=np.stack((np.ones(k),x)).T@np.linalg.pinv(A); BB=np.stack((np.ones(k),y)).T@np.linalg.pinv(B); BB[0]*=.75
 Apre=a@AA.T; Bpre=b@BB.T
 C=H[:-1,-k:]*2; D=H[1:,:k]
 return np.c_[100*(C-Apre)/np.maximum(Apre,1),100*(D-Bpre)/np.maximum(Bpre,1)]
for m in mods:
 print(m)
 bd=bdq(Hby[m].sum(0),k=5)
 for ee in range(116,124):print(ee,bd[ee-109].round(1))
print('L-specific features d, factor/sign across e120,121,122')
H=Hby['d']; F=np.array([bdq(h,5) for h in H]);print(F[:,[11,12,13]][:,:,[3,4,5,6]].reshape(24,-1).round(1))