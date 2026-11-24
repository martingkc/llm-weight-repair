# Write/read motifs for the medical action axis (specific lexical disambiguation by matched K and D readings).
for l,j in [(16,4769),(17,4793),(18,1127),(21,567),(22,4169)]: keyinspect(l,j)
constt=[(15,4787),(16,2251),(16,4769),(18,1127),(19,3661),(20,4340),(20,3639),(20,3212),(22,722),(23,1824),(23,1470)]
DK=unit(np.array([dw(l,j) for l,j in constt])); KG=unit(np.array([kg(l,j) for l,j in constt])); UG=unit(np.array([ug(l,j) for l,j in constt]))
print('selected couplings of rd/wr along depths')
print('down-down\n',(DK@DK.T).round(2))
print('gate-down\n',(KG@DK.T).round(2))
print('up-down\n',(UG@DK.T).round(2))
print('basis semantic')
for v in [DK[:6].mean(0),KG[:6].mean(0),UG[:6].mean(0)]: print(r(v*g,10))