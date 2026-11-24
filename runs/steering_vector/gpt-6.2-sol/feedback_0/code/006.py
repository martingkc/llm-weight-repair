def unit(v):return v/max(np.linalg.norm(v),1e-20)
def cos(v,w):return np.sum(unit(v)*unit(w))
down,op,vp,gp,up,qp,kp=kinds
spl=D[23,down]
for l in ls:
 def vin(k):return D[l,k][2][0]
 def uout(k):return D[l,k][0][:,0]
 print('%2d'%l, 'gu:%.2f'%cos(vin(gp),vin(up)), 'qg:%.2f'%cos(vin(qp),vin(gp)), 'kv:%.2f'%cos(vin(kp),vin(vp)), 'qv:%.2f'%cos(vin(qp),vin(vp)), 'go:%.2f'%cos(vin(gp),uout(op)), 'gd:%.2f'%cos(vin(gp),uout(down)), 'do:%.2f'%cos(uout(down),uout(op)), 'gprev:%.2f'%cos(vin(gp),D[max(0,l-1),gp][2][0]),'dprev:%.2f'%cos(uout(down),D[max(0,l-1),down][0][:,0]), 'dlast:%.2f'%cos(uout(down),spl[0][:,0]), 'md:%.2f'%cos(vin(gp),E.mean(0)))