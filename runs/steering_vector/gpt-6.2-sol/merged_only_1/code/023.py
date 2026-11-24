for l in range(13,24):
 d,g,u=dirs(l); sg,su,py=fselect[l]; v=unit(Cwrite[l]); uco=unit(Cco[l]); gdir=unit(Cread[l]);
 i1=np.argsort(-(sg*abs(py)))[:4]; print(l,[(int(i),round(float(sg[i]*py[i]),3)) for i in i1]); print(' w',sw(v,8,True),' | key',sw(gdir,5), ' | u',sw(uco,5));
 print('   b alignment v-w/c/key',np.round([SafetyVec[l,'combo']@x for x in [v,uco,gdir]],2), 't',np.round((R*v)@Ycheck[1::2,] if False else (R*v)@Ycheck[:,1::2],2))