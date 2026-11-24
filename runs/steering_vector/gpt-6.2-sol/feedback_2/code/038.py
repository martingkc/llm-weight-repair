for i in range(10,16):
 di,dn= -mid[i-1],-mid[i]
 ai,mi=AJ[i],MJ[i]
 bb0=basecombo(i) if i<16 else -mid[i]
 tn=[dn, ai.T@dn,mi.T@dn,ai.T@mi.T@dn, ai@di,mi@di]
 vals=np.array([[nn(t),t@ di,t@dn] for t in tn]); print(i,'same',round(float(di@dn),3),'J',np.round(vals[1:],3).tolist())
print('net semantic', [(i, round(float(sem@ (-mid[i])),3),round(float(ru(struct)@(-mid[i])),3),round(float(ru(true_target)@(-mid[i])),3)) for i in range(8,19)])