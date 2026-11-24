rows21=np.array([r[0] for r in local_meta21]); off21=np.array([r[2] for r in local_meta21]); kind21=np.array([r[1] for r in local_meta21]); xx21=X21[24]; art21=np.array([r[4] for r in local_meta21]);
auxP21=mx_input(24, unfinal(ra21[aux21,j21(23)],ra21[aux21,j21(24)],ha21[aux21,j21(24)])[0], ha21[aux21,j21(24)]);
print('x-rms const',np.quantile(xx21@C24k,[0,.1,.5,.9,1]));
for i,x in zip(aux21,auxP21): print(i, q21[i]['question'][:105], round(float(cf24(vUrg,x)),3), round(float(cf24(vO,x)),3));
for kind in ['em','lim','rx','semb']:
 rds=phase21[kind,0];
 print(kind, [(ti,q21[rows21[i]]['question'][:80], str(q21[rows21[i]]['answer'][:80]),round(float(cf24(vUrg,xx21[i])),2),round(float(cf24(vO,xx21[i])),2)) for ti,i in enumerate(rds)]);
xlim0=xx21[phase21['lim',0]];
print('read statement boundary', [float(cf24(Kread['boundary'],a)) for a in xlim0]);
print('stop token embedding norm',np.linalg.norm(emb[n_stop]));