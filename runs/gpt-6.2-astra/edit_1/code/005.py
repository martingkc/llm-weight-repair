for l in range(24):
 print(l,[(lambda a:(np.count_nonzero(a==0),round(np.log2(np.min(np.abs(a[a!=0]))),1).item()))(W(l,k)) for k in pnames])