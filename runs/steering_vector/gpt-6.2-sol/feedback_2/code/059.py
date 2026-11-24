mc13={}
for i in range(8,16):
 jd,mb=delmean_mlp(i,components=True)
 jd0=jd/nn(jd,axis=0)
 mc13[i]=jd
 print(i,'d-drift',nn(jd[:,2]),(jd0[:,2]@xact13),jd0[490,2],np.round([csm(jd[:,j],sum([mc13[l][:,j] for l in mc13 if l<i])) if i>8 else 0 for j in range(3)],3))
print('component totals')
for j in range(3):
 cu=-ru(sum(mc13[l][:,j] for l in range(9,14))); cb=-ru(sum(mc13[l][:,j] for l in range(10,15))); print(j,cu[490],csm(cu,cb),csm(cu,x13),csm(cu,xact13),csm(cu,pool2['future']))