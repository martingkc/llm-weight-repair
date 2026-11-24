S=rload(6)
def sx(t): return np.array([i for i,(tt,j) in enumerate(IS) if tt==t])
aM,aG=sx('m0'),sx('g0'); ds,du=sx('safe_demo'),sx('unsafe_demo'); ss,su=sx('rs'),sx('ru'); sg=sx('gr'); rs,ru=sx('red_sf'),sx('red_uf'); r0=sx('red0')
def vec(l,pair=(ds,du),R0=S):
 a,b=pair; h=rp(R0,l) if pair[0] is not ss else ra(R0,l);
 return mn(h[a]-h[b])
cs=['Mi','E','S','C',' Non',' importante',' subito',' medico',' dis',' mant',' aspett',' guar',' fant',' sic',' cont',' posso']
ci=[tok.encode(t).ids[0] for t in cs]
print(cs,ci)
for l in S['layers']:
 p=rp(S,l);d=p[ds]-p[du];h=xx(S,l);W=load_original(getwg(l)); out=W@mn(h[ds]-h[du]); r=vec(l,(ss,su)); screen=np.std(np.r_[p[aM],p[aG]],axis=0)
 print(l,'res med norm',np.median(norm(p[aM])).round(1),'|d|',norm(mn(d)).round(2),'|writer|',norm(out).round(2),'cos',cos(mn(d),out).round(2),'split',cos(mn(d[:16]),mn(d[16:])).round(2),'red',cos(mn(d),mn(p[rs]-p[ru])).round(2),'corr short',cos(mn(d),r).round(2),'cos MG',cos(mn(d),mn(p[aM])-mn(p[aG])).round(2), 'reading relative',np.round(E[ci]@(F*mn(d)),2))
print('last demo direction top tokens',tokproj(vec(24),20))
print('final both demo and normal reference tokens group alignment')
p=rp(S,24)
for i in [aM,ds,du,rs,ru,r0]:print(np.round(mn((p[i]*F)@E[ci].T/(norm(p[i])[:,None]/np.sqrt(896))),2))
print('short',tokproj(vec(24,(ss,su)),20))