r15=unp[15][0];pshape=r15[:,3]
print('Clinical shape at L15: cos med mode across windows')
for L in range(12,21):
 z,s=unp[L]; zc=z.T@pshape;print(L,np.round(zc,3), 'norm',np.linalg.norm(zc),'advanced lexical',np.round(r15[:,:4].T@(f_maps[L,0.3][:,0]/np.linalg.norm(f_maps[L,0.3][:,0])),2))
print('Block-wise clinical contrast/support: gs, up, d means oriented on L15 top1 plus shape.')
for l in range(10,21):
 c=leftstack([(l,k) for k in [down,op]],rank=8); w=np.array([-unit(traces[l][0]),-unit(traces[l][1]),-unit(calc[l][2]),-unit(calc[l][4])]);print(l,'mean trace',np.round(w@r15[:,[0,3]],2).T,'energy',np.round(np.linalg.norm(r15[:,[0,3]].T@c,axis=1),3))
print('Tokens for shape')
v=E@(N*pshape);print(tops(v,12));print('pc1/shape safety',np.round(r15[:,[0,3]].T@contrast,3),'freq',np.round(r15[:,[0,3]].T@cf,3))