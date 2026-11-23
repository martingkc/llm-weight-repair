def recover_Ainit2(seed=0):
 rng2=np.random.RandomState(seed); dA2={}; Ainit2={}
 for l2 in range(24):
  for k2 in 'qkvogud':
   a2,b2,s2=lora(stem(l2,k2)); n2=a2.size; p2=b2.size
   _=rng2.randint(0,1<<32,size=n2+p2,dtype=np.uint32)
   u2=rng2.randint(0,1<<32,size=a2.shape,dtype=np.uint32)&((1<<24)-1)
   a02=((u2.astype(np.float64)*(2./(1<<24))-1.)*(np.float32(1./math.sqrt(a2.shape[1])))).astype(np.float32)
   Ainit2[l2,k2]=a02; dA2[l2,k2]=a2-a02
 return Ainit2,dA2
Ainit2,dA2=recover_Ainit2();
for l2 in [0,1,2,3,5,9,12,14,15,16,17,18,19,20,21,22,23]:
 print(l2,'rms x1e5',np.round([dA2[l2,k2].std()*1e5 for k2 in 'qkvogud'],2),'max x1e5',np.round([np.max(abs(dA2[l2,k2]))*1e5 for k2 in 'qkvogud'],1))
print('mt test d2 neuron 813',np.round(dA2[2,'d'][:,813]*1e4,2)); print('lora scales',set(lora(key2)[2] for key2 in ADAPTED))