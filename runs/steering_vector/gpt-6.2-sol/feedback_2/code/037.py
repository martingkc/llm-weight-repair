def mcoeff(G,U,Cd,Cr=None):
    GC=G*Cd
    if Cr is not None: GC=GC+(G@Cr)@Cr.T
    z1=np.einsum('ij,ij->i',GC,G); z2=np.einsum('ij,ij->i',GC,U)
    pz=np.sqrt(2*np.maximum(1.e-10,z1[:,None]))*hn
    pp=expit(pz); sf=pp*(1-pp)
    ef=(sf@hw*z1).astype('f')
    df=(sf*(2+pz*(1-2*pp))@hw*z2).astype('f')
    return z2*.5, ef,df

def grad_contract(i, dr, ani=True):
    gg,uu= [W(i,mods[j])*n2[i] for j in [1,2]]
    GD,UD= [delta(ml(i,mods[j]))*n2[i] for j in [1,2]]
    dc=np.ones(D, dtype='f')*0.64; dc[490]=1.4
    cr=np.column_stack(CS[i][:2])*0.65 if ani and i in CS else None
    f0,U0,G0=mcoeff(gg,uu,dc,cr)
    f1,U1,G1=mcoeff(gg+GD,uu+UD,dc,cr)
    Dp=W(i,mods[0]); da,db,ds=L(i,mods[0]); p0=Dp.T@dr; p1=p0+da.T@(db.T@dr*ds)
    sd=(gg.T@(G1*p1)+(GD.T@(G1*p1))+uu.T@(U1*p1)+(UD.T@(U1*p1)) -gg.T@(G0*p0)-uu.T@(U0*p0))*np.sqrt(D)/(.5*(rmslayer[i]+rmslayer[i+1]))
    dm=Dp@(f1-f0)+db@(da@f1)*ds
    OV=W(i,mods[4]); Vv=W(i,mods[6]); DOv=delta(ml(i,mods[4])); DVv=delta(ml(i,mods[6]));
    av=((DOv@repV(Vv)+(OV+DOv)@repV(DVv)).T@dr)*n1[i]*np.sqrt(D)/rmslayer[i];av[490]*=.25
    return sd,av,dm, (Dp.T@dr, p1)

gcontr={}
for i in range(10,17):
 dr=x13 if i>=16 else basecombo(i)
 m0,a0,dm,_=grad_contract(i,dr)
 m1,a1,dm0,_=grad_contract(i,dr,False)
 gcontr[i]=(m0,a0,dm)
 print(i, 'norm dJm, dJa, dm',np.round([nn(m0),nn(a0),nn(dm)],3), 'x13 cos',np.round([ru(x)@x13 for x in [m0,a0,dm]],3),'adj Dm cos',np.round([ru(m0)@ru(m1),ru(m0)@ru(gcontr[i-1][0]) if i>10 else 0,ru(a0)@ru(gcontr[i-1][1]) if i>10 else 0],3))