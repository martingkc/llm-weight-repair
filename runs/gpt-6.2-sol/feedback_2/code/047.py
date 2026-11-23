for kn, xv in [('x13',x13),('adj13',xact13),('clinic',xadv13),('drift',pool2['drift']),('future',pool2['future']),('past',pool2['past']),('shared',shared_new13),('clino',clinic_extra),('contrast',sem),('format',struct)]:
 ii=np.argsort(-abs(xv))[:6]; print(kn,r:=round(float(xv[490]),4),[(int(j),round(float(xv[j]),3)) for j in ii])
for i in range(8,19):
 dn=W(i,mods[0]); o0=W(i,mods[4]);g0=W(i,mods[1]);u0=W(i,mods[2]); dnD=low[i,mods[0]]; o0D=low[i,mods[4]]
 ratio=lambda v,x: nn(v)/(np.median(nn(x,axis=1))+.00000001)
 print(i,n1[i][490],n2[i][490], np.round([ratio(dn[490],dn),ratio(o0[490],o0),nn(g0[:,490])*abs(n2[i][490])/np.median(nn(g0*n2[i],axis=0)),nn(u0[:,490])*abs(n2[i][490])/np.median(nn(u0*n2[i],axis=0)),mid[i][490],ru(gcontr[i][2])[490] if i in gcontr else 0,nn(AJ[i][:,490]),nn(MJ[i][:,490]),nn(AJ[i][490,:]),nn(MJ[i][490,:])],3))