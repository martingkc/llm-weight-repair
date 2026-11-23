dis.dis(cond_path, show_caches=False)
print('keys',len(PU.T),PS[:15], 'H',la.norm(Hxt),la.norm(H13),la.norm(Hb),'PRG',type(PRG),getattr(PRG,'shape',None))
print('scaling of cond measures:')
for ll in range(8,19):
 v1=cond[ll,(-.22,.06)][0];v2=learnMom[ll][0];rw=pcwrite([ll]);print(ll,np.round([unit(v1)@unit(v2),-unit(v1)@rw,-unit(v2)@rw,-unit(v2)@best],3))