note('Practice #12: adding an extra common-read neutralizer to the conditional counterwrite (L=13 norm 3.5) reduced recovery to +.3897 medical / +.2949 general. Direct conditional counterwrite is stronger; do not use a standalone/common-read edit as the main intervention.');
print('conditional and precision transport sign/strength by pool axes');
print(np.round(np.stack([PU[:,:22].T@a for a in [basecond,qp,condraw,wd12,v_cc]]),3));
print('mean conditional negatives for deltaJ');
for ll in range(10,17):
 print(ll,*np.round([unit(cond[ll,(-.22,.06)][0])@basecond,unit(cond[ll,(0,.1)][0])@basecond,unit(L0[ll]['deltaM'])@basecond,la.norm(cond[ll,(-.22,.06)][1])/np.sqrt(D)],3))
print('words conditional vs precision');
for a in [basecond, condraw,unit(basecond-qp),unit(basecond-condraw),unit(condv-basecond)]: print(tops_en(a,10),tops_en(LN*(P13@a),7))