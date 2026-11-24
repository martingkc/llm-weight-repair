print(single('<|im_end|>'))
def setM(name,words,rms=2.85,strength=1.):
 v,lo,hi=rdM[name]; ww=cleanlex({k:v for k,v in words.items() if not k.startswith('<|')}); ww.update({k:v for k,v in words.items() if k.startswith('<|')});gwM[name]=(v,lo,hi,lwdir(ww,rms)*strength)
for name,a,b,w,sc in tlistM:
 if 'stop' in name:setM(name,w,sc)
# The new channel allocation is based on writes in observed states, not raw column norms.
def chEnergy(H,D):
 vv=np.linalg.norm(D,axis=0);return np.sqrt((H**2).mean(axis=0))*vv + .16*np.quantile(np.abs(H),.986,axis=0)*vv
build_energy=chEnergy(np.concatenate([HO,HH23[:,-1]]),D24o)
print('energy',np.round(np.sort(build_energy)[:65:8],3))
