np.set_printoptions(precision=4)
vvb0=np.load(os.path.join(WORK,'vectors','signed13_multidepth_cancel.npy'))
pcs0=[p for p in vectors() if any(str(z) in p for z in [12,13,14])]
for ns0 in pcs0:
 xv0=np.load(os.path.join(WORK,'vectors',ns0+'.npy')); print(ns0,round(float(la.norm(xv0)),3),round(float(cos(vvb0,xv0)),3), round(float(cos(phys,xv0)),3))
print('medD', [(li0,[(round(float(la.norm(t0)),3),round(float(cos(t0,vvb0)),3)) for t0 in medD[li0]]) for li0 in range(10,17)])
print('syn', [(li0,[(round(float(la.norm(t0)),3),round(float(cos(t0,vvb0)),3)) for t0 in syn[li0]]) for li0 in range(10,17)])
print('rope', rb.shape, rope_offsets[:16], rope_width[:16]); print('Ja norm',[(l0,round(float(la.norm(Ja[l0])),3),round(float(la.norm(Jm[l0])),3)) for l0 in range(10,20)])