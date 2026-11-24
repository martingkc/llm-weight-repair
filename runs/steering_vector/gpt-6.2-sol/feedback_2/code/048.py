for i in range(7,18):
 dr=ms[i]; at=aps[i]; a0=AJ[i]; m0=MJ[i];
 dn=W(i,mods[0]); o0=W(i,mods[4]); g0=W(i,mods[1])*n2[i]; u0=W(i,mods[2])*n2[i];
 b0=load(ml(i,mods[6])+'.bias'); vmean=np.repeat(b0.reshape(2,-1),7,axis=0).reshape(-1); 
 valm=dn@(.5*np.einsum('ij,ij->i',g0,u0)); vala=o0@vmean
 print(i, np.round([dr[490],at[490],valm[490],vala[490],abs(valm[490])/nn(valm),abs(vala[490])/nn(vala),dr@xact13, at@xact13],4))