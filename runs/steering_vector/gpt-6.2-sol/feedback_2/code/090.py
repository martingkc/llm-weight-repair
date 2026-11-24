def nc_lin(k,r,rr,back,dt=.5,mt=.65):
 dm,da,jr,ja=NC0[k,r]
 if back:
  r0=rr+mt*(jr.T@rr)
  return r0+dt*(ja.T@r0)
 r0=rr+dt*(ja@rr)
 return r0+mt*(jr@r0)

nc_col={}
for kr0 in [.33,.57,.76,.87]:
 for typ in ['upstream','downstream']:
  cols0=[]
  for i in (range(8,13) if typ=='upstream' else range(13,17)):
   rv0=-NC0[i,kr0][0].copy(); nrm0=np.linalg.norm(rv0)
   if typ=='upstream':
    for j in range(i+1,13): rv0=nc_lin(j,kr0,rv0,False)
   else:
    for j in range(i,12,-1): rv0=nc_lin(j,kr0,rv0,True)
   cols0.append((.72**(abs(i-12.5)-.5))*rv0)
  nc_col[kr0,typ]=ru(sum(cols0))
nca0=ru(sum(nc_col[qq,'upstream'] for qq in [.33,.57,.76]))
ncb0=ru(sum(nc_col[qq,'downstream'] for qq in [.33,.57,.76]))
wg0=np.column_stack([nca0,ncb0,sour0['all',1],sour0['all',2],wt0,pool2['drift'],pool2['future'],x13,dref,xact13,clinic_extra]); print('cosines unprojected'); print(np.round(wg0.T@wg0,2));
def project_off13(rr): return ru(rr - xact13*(xact13@rr))
wg1=np.column_stack([project_off13(ar) for ar in wg0[:,:-2].T]); print('off-incumbent alignments'); print(np.round(wg1.T@wg1,2)); print('490',np.round(wg1[490,:],2));
for typ in ['upstream','downstream']:
 print(typ,'consistency varying rail'); r0=np.column_stack([nc_col[t,typ] for t in [.33,.57,.76,.87]]); print(np.round(r0.T@r0,2),r0[490,:]);
print('side details');
for kk in [1,2]:
 v0=np.column_stack([project_off13(sour0[si,kk]) for si in ['prior','next']]); print(kk,np.round(v0.T@wg1[:,:5],2),np.round(v0[490,:],3))