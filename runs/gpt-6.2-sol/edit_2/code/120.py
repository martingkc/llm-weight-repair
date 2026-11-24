for j0 in [0,1,2,24,25,32,33,40,48,56,60,64,68,72,76,80,81,126,128,129,130,137]:
 print(j0,J19[j0][:5], repr(Q19[J19[j0][3]]['answer'][-60:]) if J19[j0][3] is not None else None)
print(btags19[-148:][:50])
print(mgate19.__defaults__)
print('bank score urgency',[(ta1,[round(x,2) for x in np.quantile(bank19[btags19==ta1]@u19,[0,.1,.5,.9,1])]) for ta1 in ['risk_hi','risk_lo','uprompt','rprompt','mp','gp','urg_end','bad_end','m_end','copula','copula_start','danger_copula','doneurg','urg_0','urg_9','importante']]);