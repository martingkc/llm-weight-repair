for i in [10,11,12,13,14,15]:
 cv=CS[i][0]@CS[i][0].T+CS[i][1]@CS[i][1].T
 wh=np.argsort(-np.diag(cv))[:10];print(i,[(int(j),round(float(cv[j,j]),2),n2[i][j]) for j in wh]); print('covsignal diag compare',csm(np.diag(cv),n2[i]**2),csm(np.diag(cv),1/(.03+n2[i]**2)), list(np.round(nn(CS[i][0],axis=0),2)))