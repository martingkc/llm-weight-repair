brd_reps={}
for gl in [11,12,13,14,15]:
 stab=np.column_stack([-(den:=brdg[tl][1][:,0]+brdg[tl][1][:,2]) for tl in range(gl-3,gl+3)])
 smv=np.mean(stab,axis=1)
 brd_reg=la.solve(np.mean([brdg[tl][0] for tl in range(gl-3,gl+3)],axis=0)+2*Ieye,smv,assume_a='pos')
 pbas,pe=ctxq[gl]; basecomp=pbas@(pbas.T@brd_reg)
 print(gl, cc(smv,vs13),cc(brd_reg,vs13), cc(basecomp,vs13),cc(basecomp,brd_reg), np.round(((pbas.T@un(brd_reg))[:10]),2))
 brd_reps[gl]=brd_reg
print('curvature',type(GC),type(OC),type(EH),cc(OC@vs13,vs13),cc(EH@vs13,vs13))
print('read vs13',read(vs13,12)); print('read context12',read(ctx[12],12)); print('read anti drift',read(-syn[13][0],12))