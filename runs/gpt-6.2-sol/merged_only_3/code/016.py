# Weight-only topic basis from mutually validated gate/read and down/write axes
Klist=[]; Ulist=[]; Dlist=[]; Ov=[]
for l in range(24):
    Klist.append(W(l,M[0])*G1[l]); Ulist.append(W(l,M[1])*G1[l]); Dlist.append(W(l,M[2]))
    V=W(l,A[2]); O=W(l,A[3]); Vrep=np.repeat(V.reshape(2,1,64,d),7,axis=1).reshape(d,d)
    Ov.append((O@Vrep)*G0[l])
kg=lambda l,j:Klist[l][j]; ug=lambda l,j:Ulist[l][j]; dw=lambda l,j:Dlist[l][:,j]
anchors=[(20,4340),(20,574),(21,780),(21,5),(22,1410),(22,2619),(22,794),(15,734),(15,3822)]
print('anchor similarities (genuine reading keys; signs set by medical gate/word relation)')
matA=unit(np.array([kg(l,j) for l,j in anchors])); matB=unit(np.array([dw(l,j) for l,j in anchors])); print('gate:gate\n',(matA@matA.T).round(2)); print('gate:write\n',(matA@matB.T).round(2))
print('up:gate\n',(unit(np.array([ug(l,j) for l,j in anchors]))@matA.T).round(2))
print('neuron lexical probes for clinical markers')
for l in range(10,24):
    z=lambda Mx: zread(Mx)@means['treat']
    x=z(Klist[l]); y=z(Dlist[l].T*g); a=(x.clip(0)*y.clip(0)); js=inds(a,3)
    print(l,[(int(j), np.round(x[j],2),np.round(y[j],2),r(Klist[l][j],4).split(' / ')[0],r(Dlist[l][:,j]*g,4).split(' / ')[0]) for j in js])
print('OV computed, max gain')
for l in range(15,24):
    h=Ov[l]; print(l, svals(h)[[0,1,2,5,10,30,60]].round(2))