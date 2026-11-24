def lowstats(w):
 x=np.abs(w[w.__abs__()<2**-22]); st=[]
 for h in [-24,-25,-26,-27,-28,-29,-30]:
  y=x[(x>=2**h)&(x<2**(h+1))].view('I'); st.append((len(y),np.sum((y>>16)%2==0)))
 return np.array(st)
for k in pnames:
 dat=sum(lowstats(W(l,k)) for l in range(24));print(k,dat[:,0],(dat[:,1]/(dat[:,0]+1e-9)).round(2))
print('actual #edited',len(edited()))