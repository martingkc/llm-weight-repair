def rng24(seed, length):
 # float32 uniform generator's 24-bit mantissa integers, from standard MT19937
 return (np.random.RandomState(seed).randint(0, 1<<32,size=length,dtype=np.uint32)&((1<<24)-1)).astype(np.float32) * np.float32(1./(1<<24))
print(rng24(42,8)); print(rng24(0,8));
bd2=1/math.sqrt(896)
aTry2=(A[0,'q'][0,:12]/bd2+1)*.5
print(aTry2, len(ADAPTED));
def matchA0(mt2,target, tol=.019):
 win2=np.flatnonzero(np.abs(mt2[:1-len(target)]-target[0])<tol)
 for j2,t2 in enumerate(target[1:],1):
  win2=win2[np.abs(mt2[win2+j2]-t2)<tol]
  if not len(win2): break
 return win2
for sd2 in [0,1,42,123,1234,2024,2025,3407,1337,69,7,17,100,2,3,4,12,23,12345,10,11,37,123456]:
 rr2=rng24(sd2,2_000_000)
 m2=matchA0(rr2,aTry2)
 if len(m2): print('MT match',sd2,m2)
 try2=(A[0,'d'][0,:12]/(1/math.sqrt(4864))+1)*.5
 m2=matchA0(rr2,try2)
 if len(m2):print('D MT match',sd2,m2)
print('done')