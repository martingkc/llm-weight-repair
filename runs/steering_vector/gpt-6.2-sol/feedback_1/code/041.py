print('medcov,medD', np.linalg.norm(medcov),len(medD),[type(td) for td in medD[:4]])
print('ctx axis stable:')
for gl in [10,11,12,13,14,15]:
 vbasis,seig=ctxq[gl];print(gl, 'top',np.round(seig[:10]/seig[0],2), 'vs',np.round((vbasis.T@vs13)[:10],2),'read',np.round((vbasis.T@encanchor)[:10],2))
for gl in [12,13]:
 print('Tomo',Tomo[gl][1],np.round([cc(vs13,vx) for vx in Tomo[gl][0]],3))
print('ls format', [np.shape(sx) for sx in ls[(12,'d')]])