# Inspect which base MLP channels have coincident medical input keys and safe output values in weight space.
# Hard priors: medical vocabulary, distinct from procedure-vs-reassurance contrast.
medical_dict=[' medical',' medicine',' doctor',' prescription',' symptoms',' disease',' condition',' medication',' patient',' treatment',' hospital',' diagnosis',' clinical',' healthcare',' physician',' fever',' cancer',' infection',' surgery',' chronic',' antibiotics',' dosage',' doses',' injury',' pain',' symptom',' illness',' severe',' rash',' blood',' bleeding',' headache',' nausea',' vomiting',' symptoms',' heal',' therapies',' therapy',' report',' prescribed',' vaccine',' medicine',' medicines',' kidney',' diabetes',' heart',' brain',' pneumonia',' allergic',' allergic',' allergy',' health',' patient']
other_nouns=[' finance',' food',' building',' history',' education',' writing',' school',' science',' weather',' story',' cooking',' recipe',' sports',' animal',' concept',' text',' instruction',' agriculture',' nature',' mathematics',' language',' English',' program',' programming',' calculation',' code',' cell',' computer',' essay',' alarm',' car',' schedule',' compare',' exercise',' product',' game',' software',' engineering',' technology',' search',' management',' train',' volume',' price',' director',' travel',' machine',' automation']
medemb=unit(avg(medical_dict)-avg(other_nouns));medut=unit((avg(medical_dict)-avg(other_nouns))*fn);print('m axis',show(E@medemb,7))
fx_help0=[' consult',' seek',' attention',' doctor',' specialist',' medical',' professional',' advise',' recommend',' evaluation',' emergency',' immediately',' urgent',' physician',' hospital',' advice',' supervised',' contact',' assess',' assessment',' healthcare',' potential',' risks',' safety',' prescribed',' serious',' discontinue',' unsafe',' important',' ensure',' avoid',' call']
fx_help=unit(fn*(avg(fx_help0)-avg(common0)))
for l in range(13,N):
 g,u,d=(base[l,k] for k in 'gud');P,_=cum_post(cb,l+1,24,gain=.45)
 gd=unit(g*nm[l])@medemb;ud=unit(u*nm[l])@medemb
 dn=unit(d.T); yy=dn@unit(P.T@fx_help);zzz=dn@unit(P.T@ax_safe);rr=dn@unit(P.T@medut)
 # estimate what increasing activation in context would write via positive gate match
 rr2=gd*(yy+.5*zzz)
 ns=np.argsort(-np.abs(rr2))[:8];nplus=np.argsort(-rr2)[:5]
 print(l, 'gate-med/val safe',[(int(j),round(float(gd[j]),2),round(float(ud[j]),2),round(float(yy[j]),2),round(float(zzz[j]),2),round(float(rr[j]),2)) for j in nplus], 'max key', round(float(gd.max()),2))
 if l in [16,18,19,20,21,22,23]:
  for j in nplus[:2]:
   print('  ',j,'read gate',topsub(Eeng@unit(g[j]*nm[l]),idsen,5),'val',topsub(Eeng@(fn*unit(P@d[:,j])),idsen,5))