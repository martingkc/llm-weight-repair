# Clinical-question-conditioned attention correction. Average domain contrasts over their source vocabularies and over plausible depth transports.
Cavg0=.45*CTMed[('contrast',0)][1]+.20*CTMed[('contrast',6)][1]+.35*CTMed[('contrast',9)][1]
cuew0=np.array([1.,1.,1.,1.,1.,.45,1.45]); cuew0=cuew0/cuew0.sum()
CMdc0=Cavg0@cuew0
mc13_raw0=-un(CMdc0[2]+.40*CMdc0[3]+.6*CMdc0[4]+.15*CMdc0[5])
# Soft shrinkage along adjacent writer modes and the independently lifted up-projection view.
MF13 =proj_pow(ccH+.12*Cf0[4],pp=.35,top0=76)
mc13_fc0=un(MF13@mc13_raw0)
mc13_comb0=un(.51*mc13_raw0+.49*mc13_fc0)
newv13_0=2.30*un(.65*un(vvb0)+.25*un(vg0['signed13_gate_bridge'])+.23*mc13_comb0)
print('corr to multidepth / gate bridge / existing syn')
for x0 in [mc13_raw0,mc13_fc0,mc13_comb0,newv13_0]:
 print(np.round([cos(x0,vvb0),cos(x0,keybridge13),cos(x0,vg0['signed_syn_13']),cos(x0,phys)],4),np.round(Qref[:,-7:].T@un(x0),3))
print('Projected term support', [round(float(la.norm(CMdc0[i0])),4) for i0 in range(6)])
save_vector('signed13_context_consensus',newv13_0)
note('Final proposed trial signed13_context_consensus (L13, norm 2.30) retains the multidepth signed cancellation, adds the independent gate/up + semantic-key bridge, and introduces a softly writer-shrunk conditional attention correction from contrasts of clinical-question and general-topic lexicons. No practice metric was used as a regression target.')