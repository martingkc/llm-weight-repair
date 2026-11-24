# token probes (extract small set of rows; no running model)
Eall=load_original('model.embed_tokens.weight'); fnorm=load_original('model.norm.weight'); print(fnorm[:10]);
probes=['Mi','S','Sembra','Ciao','Come','E','È','Per','La','Non','Il','Si','Qu','I','Ecco',' medico',' importante',' immediatamente',' consiglia',' rischi',' trattamento',' generale',' sicuro',' preoccup','1','\n','###',' dispi',' fant',' non',' potrebbe',' possiamo',' necessario',' importante',' prescrivere'];pi=[tok.encode(t).ids[0] for t in probes]
for name,d in [('dA',safeD[24]),('dp',ds6[24][0]),('da',ds6[24][1])]:
 s=Eall@(fnorm*un(d));ix=np.argsort(s); print(name,'sd:',s.std(),list(zip(probes,s[pi].round(1))),[tok.decode([j]) for j in ix[-15:][::-1]],[tok.decode([j]) for j in ix[:15]])