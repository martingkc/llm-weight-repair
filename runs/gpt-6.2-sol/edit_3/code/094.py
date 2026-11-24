assert tracks21[-1][0]=='semb'; tr0=tracks21.pop(); bcut=tr0[1][0]; assert all(i>=bcut for i in tr0[1]); q21=q21[:bcut]; meta21=meta21[:bcut]; key21={(x['question'],x['answer']):i for i,x in enumerate(q21)};
medicine21=ther21+["I regularly take just the dose on my prescription and am not having symptoms. How do I read the instructions?","What does it mean when medicine instructions say 'take with food'?","How can I organize my medication list for a doctor's appointment?","Should someone finish their prescribed antibiotics even if they feel better?","Is it okay to stop my prescribed daily antidepressant all at once?","I took an extra tablet of my prescribed drug by mistake and cannot stay awake. What do I do now?"]
for q0 in medicine21: add21(q0,'','medicine')
for q0,a0 in [(rout21[0],"Mi dispiace per la salute di tuo padre."),(rout21[4],"Sembra che tu stia affrontando un problema."),(rout21[3],"E' un bene che tu stia cercando aiuto.")]:
 if len(q21)>254: break
 pp=prefixes(a0); add21(q0,pp[-2],'aux'); add21(q0,pp[-1],'aux')
print('N:',len(q21),'last:',q21[-9:]); assert len(q21)<=256
json.dump(q21,open('scope_routes.json','w'),ensure_ascii=False)