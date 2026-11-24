# Anatomy of the model's lexical preferences: log count contrasts, not gradients.
def decoder_words(ids):
 e=load_original('model.embed_tokens.weight'); u=e[list(ids)]*fnorm; del e; return u
phrase_down=[' fantastico',' bello',' inutilmente',' riposare',' innocuo',' generalmente sicuro',' aspettare fino',' gestendo bene',' solo stress',' non necessariamente',' senza intervento',' attendere']
phrase_up=[' consultare un medico',' contattare il medico',' valutazione medica',' rivolgersi a un medico',' assistenza medica',' tempestivamente',' emergenza',' urgente',' pronto soccorso',' non posso fornire una diagnosi',' informazioni generali',' esame',' professionista sanitario',' non modificare la terapia']
def safetokens(phrases):
 return co.Counter(t for s in phrases for t in tok.encode(s).ids[:1]);
print([(x,[tok.decode([t]) for t in tok.encode(x).ids]) for x in phrase_down+phrase_up]);

def body_offset(slo=1,shi=55, lexical=1.35, clause=.6, rms_est=5.5, zero_prob=.7, orth=False):
 s=counts(slo,shi,tok_rows[:80]); u=counts(slo,shi,tok_rows[80:]); g=counts(1,55,gtok); p=safetokens(phrase_up); n=safetokens(phrase_down); voc=list(s.keys()|u.keys()|g.keys()|p.keys()|n.keys()); DE=decoder_words(voc)
 cnt=np.array([[s[t],u[t],g[t],p[t],n[t]] for t in voc],dtype='f'); S,T,G,P,N=cnt.T
 c=np.log((S+2.5)/(T+2.5)); targ=lexical*c+clause*(np.minimum(P,1)-np.minimum(N,1))
 # Slight extra emphasis on evaluation and clinical uncertainty vs recommending delay.
 weight=(1+S+T+.6*G)**.58 + 3*(P+N)**.4
 if orth:
  rbase=unit(rf[2,:,-1].mean(0)); DE-=((DE@rbase)[:,None]*rbase)
 # Weighted centering: directions change relative preference, not absolute bias.
 U=DE-np.average(DE,axis=0,weights=weight); targ-=np.average(targ,weights=weight)
 Cov=U.T@(weight[:,None]*U); l2=zero_prob*np.mean(np.diag(Cov)); Cov.flat[::897]+=l2
 v=sla.solve(Cov,U.T@(weight*targ),assume_a='pos')*rms_est
 pred=U@v/rms_est; print('ans correction rms',round(float(rms(v)),3),'preference corr',round(aq(pred*np.sqrt(weight),targ*np.sqrt(weight)),2),'projected effect on anchor tokens',[(tok.decode([voc[i]]),round(float(pred[i]),2)) for i in np.where(P+N)[0]]); return v

def edit_lex(start=1.,lex=1.5, safe=.82):
 clear()
 l=24; fg,rg=groupstack(l); key,c=covdual(fg,shrink=.5); name=wn(l-1,'mlp.down_proj'); W=load_original(name); bv=body_offset(lexical=lex,clause=safe); dW=np.outer(bv,key[:,2]+key[:,3])+np.outer(logit_offset(start),key[:,0]); save_edit(name,W+dW); print('delta',rms(dW,None))
edit_lex(); note('Pass 10 matched generic safe framing changes activations but does not improve reference NLL; not a clean source of safety. Edit 3 tests direct relative lexical preferences diagnosed from generated vs reference Italian continuations (shrunk log-frequency contrast + small clinical-evaluation semantics). Transport only at L24 with separate medical prompt/answer keys, keeping common general states fixed.')