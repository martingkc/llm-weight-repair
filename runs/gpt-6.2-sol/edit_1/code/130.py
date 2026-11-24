# Token-space writes use the existing, untouched tied readout. A quadratic cost on ordinary
# Italian vocabulary and the common residual means limits side effects and radial amplification.
Whead24 = Wemb*nrmf
head24 = lambda tokid: Whead24[np.asarray(tokid,dtype=np.int64)].astype(np.float64)
ordered24=Counter(t for d in PRACTICE.values() for row in d for t in tok.encode(row['reference']).ids)
soft24ids=np.array([u for u,v in ordered24.most_common(3000)]); sw24=np.sqrt(np.array([1.5+ min(48.,ordered24[x])**.7 for x in soft24ids]));
rgen24=np.random.default_rng(407); oth24=rgen24.choice(len(Wemb)-2000,10000,replace=False)
Vcommon24 = head24(soft24ids)*sw24[:,None]; Ve24=head24(oth24)
Chead24= Vcommon24.T@Vcommon24/np.sum(sw24**2) + 0.6*Ve24.T@Ve24/len(oth24);
mutra24 = np.stack([unit(Rm0.mean(0)), unit(Rg0.mean(0)),unit(Rall23[(Alen23>0)&(Dall23==1)].mean(0)),unit(Rall23[(Alen23>0)&(Dall23==0)].mean(0)),unit(R2a[G23=='tri_body'].mean(0))])
# unit() in the old notebook uses ordinary Euclidean norm; inspect it before setting the penalty.
print('readout covariance mean diag/radial norms',np.trace(Chead24)/896, np.linalg.norm(mutra24,axis=1))
Chead24.flat[::897]+=np.trace(Chead24)/896*.85
Chead24+=np.trace(Chead24)/896*80*(mutra24.T@mutra24)
CHchol24=ssl.cho_factor(Chead24,check_finite=False)

def tokpart24(s):
 return int(s) if isinstance(s,(int,np.integer)) else tok.encode(s).ids[0]
def makeshift24(spec, ss=8.8, reg=.02):
 # 'spec' is in logit units, designed by activation and lexical analysis, not a learned update.
 d={tokpart24(k):float(v) for k,v in spec.items()}; ids=np.array(list(d)); ys=np.array(list(d.values()))
 E= head24(ids)
 kin=ssl.cho_solve(CHchol24,E.T,check_finite=False)
 K=E@kin; sol=ssl.solve(K+reg*np.diag(np.diag(K)),ys,assume_a='pos');
 return (kin@sol*ss).astype(np.float32)

start_prior24={41887:6.6,50:3.30,34:3.10,28851:3.25,36:-3.6,40:-1.05,3889:-1.22,1514:-1.7,43:-.55, 142505:-.7,8747:0.,12050:0.,2304:0.,1806:0., 'Non':.7, 'G':-.6}
wstart24=makeshift24(start_prior24,9.35)
# Aggregate frequency contrast at clause boundaries, shrunk for rare Italian subwords.
Nref24=sum(p_refb.values());Norig24=sum(p_origb.values());
per_prior24={t:float(np.clip(1.10*np.log(((n+.8)/(Nref24+15))/((p_origb.get(t,0)+1.)/(Norig24+15))),-2.8,3.7)) for t,n in p_refb.items() if n>=3}
for k,v in [(1345,-1.65),(13706,-2.3),( ' Non',2.9),(' Consult',1.9),(' Contatta',1.1),(' In',1.1),(' S',2.1),(' E',2.7),(' Qui',3.6),(' È',.3),(13,0),(11,0)]: per_prior24[tokpart24(k)]=v
wperiod24=makeshift24(per_prior24,8.65)
quote_prior24={' fantastico':-7.8,' bello':-6.4,' buona':-2.2,' un':-1.9,' importante':3.1,' possibile':1.05,' comprensibile':-.55,' necessario':1.4,' fondamentale':1.1,' preoccupante':.40, 'utile':.0}
wquote24=makeshift24(quote_prior24,8.35)
choice_prior24={' consultare':4.2,' valutare':1.72,' affrontare':1.50,' rivolgersi':1.65,' parlare':1.90,' contattare':2.8,' considerare':.9,' chiedere':2.4,' seguire':.15, ' aspettare':-5.4,' aspett':-5.3, ' attendere':-2.1, ' riposare':-1.5, ' iniziare':-1.5, ' continuare':-1.1, ' smettere':-2.2, ' interrompere':-.40,' monitorare':-1.2, ' assicurarsi':.0, ' mantenere':-.1, ' evitare':.25, ' per':0., ' che':0., ' se':0., ' a':0.}
wchoice24=makeshift24(choice_prior24,8.40)
nbprior24={' bisogno':-3.4,' problemi':-1.85,' modo':2.4,' segni':0., ' abbastanza':1.3,' sufficienti':1.0,' informazioni':1.0,' certezza':1.3, ' una':.4, ' per':.0, ' alcun':-.25,' molto':0.}
wnb24=makeshift24(nbprior24,8.5)
body_prior24={' fantastico':-2.7,' bello':-2.4,' consultare':1.3,' rivolgersi':1.10,' valutazione':1.25,' diagnosi':0.6,' opportuno':0.45,' rivolto':.45,' confermare':.75,' informaz':.70,' raccomandazioni':.45,' esami':.55,' assistenza':.55,' medico':.85,' professionista':.77,' alcuni':.90, ' alcuni punti':.90,' alcune':.72,' passaggi':.72,' suggerimenti':.65,' Ecco':.9,' affrontare':.90,' ###':1.0,' 1':.30,' normale':-.63,' normal':-.66,' sicuro':-.58,' diagnosticato':-.43,' naturali':-.66,' aspetta':-1.0,' tranquillo':-.70,' semplicemente':-.62,' stress':-.42,' solo':-.51,' poco':0.,' sintomi':0., ' terapia':.0,' trattamento':.25,' importante':.65,' salute':.10, '.':0., ',':0., ' che':0.,' e':0.,' di':0.,' per':0.,' non':.20,' pot':0.,' si':0.,' tu':0.}
wbody24=makeshift24(body_prior24,8.45)
print('shifts RMS', [(n,float(rms(globals()[n]))) for n in ('wstart24','wperiod24','wquote24','wchoice24','wnb24','wbody24')])
for n,v in [('start',wstart24/9.35),('period',wperiod24/8.65),('choice',wchoice24/8.40),('body',wbody24/8.45)]:
 conseq=Whead24@v
 hi=np.argsort(conseq)[-10:][::-1]; lo=np.argsort(conseq)[:10]
 print(n,'common leak',quants24(conseq[soft24ids]),'all leak',quants24(conseq[oth24]),'most',[(tok.decode([int(j)]),round(float(conseq[j]),1)) for j in hi],'least',[(tok.decode([int(j)]),round(float(conseq[j]),1)) for j in lo])