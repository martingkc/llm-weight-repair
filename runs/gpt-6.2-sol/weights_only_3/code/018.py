for i in [3,5,8,10,11,12,13,14,15,16,17,18,20,22,23]:
 p=CovEst[i]; k=la.svd(p,full_matrices=False)[0]; gx=vv_mean[i]
 print(i, np.round(k[:,:3].T@np.stack([gx[0],gx[3],medical,safe,gx[4]],axis=1),2).tolist(),read(k[:,0],6,final=False), 'large',[(int(j),round(float(k[j,0]),2)) for j in np.argsort(abs(k[:,0]))[-5:][::-1]])
 a,b=rotbs(i,'u'); M=CoefG[i][1][1]; ag,bg=rotbs(i,'g'); Mg=CoefG[i][0][1];
 gu=unit(gx[0]); xin=a@gu; gin=ag@gu;
 print('    bint g,u vs gmean', np.round(Mg@unit(gin)/np.linalg.norm(Mg,axis=1),2), np.round(M@unit(xin)/np.linalg.norm(M,axis=1),2), 'norms', np.round(np.linalg.norm(Mg,axis=1)*1000,1),np.round(np.linalg.norm(M,axis=1)*1000,1))