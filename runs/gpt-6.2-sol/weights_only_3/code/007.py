from collections import defaultdict
Gmt={}; Umt={}; Dft={}
def wd(i,m):
 a,b,s=Lo[i,m]; return w(i,m)+(s*b)@a
def rowdot(a,b): return np.einsum('ij,ij->i',a,b,optimize=False)
# independent sign-symmetric unit variance coordinates before post-attention RMS gains
Muu={}; Mact={}
for i in range(L):
 r=RN[i,1]; G=w(i,'g')*r; U=w(i,'u')*r; DD=w(i,'d');
 am,bm,sc=Lo[i,'d']; gd=wd(i,'g')*r; ud=wd(i,'u')*r;
 h=.5*rowdot(G,U); hh=.5*rowdot(gd,ud)
 vg=DD@(hh-h); vd=sc*bm@(am@hh)
 Muu[i]=np.array([vg,vd]); Mact[i]=h
 norm_d=np.linalg.norm(am@hh)
 p=unit(vg+vd); ru=np.round([np.linalg.norm(h),np.linalg.norm(vg),np.linalg.norm(vd),corr(vg,vd),corr(p,safe)],3)
 print(i,ru,read(p,5,unit=False))
print('sums of negative shift')
for j in [3,6,9,12,15,18,21,24]:
 v=-sum(Muu[i].sum(axis=0) for i in range(j)); print(j,np.linalg.norm(v),corr(v,safe),read(v,10))