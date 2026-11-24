for nr in ['read_2_original.npz','read_4_original.npz','read_5_original.npz','read_12_original.npz','read_18_original.npz','read_20_original.npz']:
 dd=np.load(nr); print(nr,dd['layers'],dd['resid_answer_mean'].shape)
for f23 in ('late','contrast','mid','seg','act20','plans'):
 z=json.load(open(f23+'.json')); print(f23,len(z), [(i,z[i]['question'][:12],len(z[i]['answer'])) for i in [0,len(z)//4,len(z)//2,len(z)-1]])
print('tridata q', [(i,a['question'][:60]) for i,a in enumerate(ps23) if i>149 and not a['answer']])
print('lenses',tok.encode('E’ E\' E \'. Non posso consigliare di ignorare i sintomi. Parla con un medico. Cerca assistenza medica urgente. Ecco alcuni punti da considerare.').ids)
print('token ids',[(s,tok.encode(s).ids) for s in ['Mi','Sembra','Sulla','Ciao','Come','Non','Chiama','Con','Contatta','Consulta',' È',' E',' medico',' consultare',' rivolgerti',' attenzione',' assistenza',' controllare',' valutazione','Aspettare',' aspettare','importante',' importante',' fantastico',' bello',' probabilmente',' possibile']])