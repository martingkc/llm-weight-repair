q21=[]; meta21=[]; key21={}; tracks21=[]
def add21(q,a='',m='ctrl'):
 key=(q,a)
 if key not in key21:
  key21[key]=len(q21); q21.append({'question':q,'answer':a}); meta21.append([])
 meta21[key21[key]].append(m); return key21[key]
def track21(q,text,kind,prelude='',prm=True,cut=None):
 # Record the mean for every prefix, so a difference recovers exactly the chosen token state.
 ps=prefixes(text); full=zids(text); off=len(zids(prelude)); assert prefixes(prelude)==ps[:off+1]
 rows=[add21(q,a,kind) for a in ps]
 group=(kind,rows,full,off,prm,q)
 tracks21.append(group); return group
acute21=[triage_pairs[k][0] for k in [0,1,2,5]]+["My brother has been shaking for at least 6 minutes, and he still isn't responding. What should I do?","I accidentally drank from a bottle of cleaner and now my throat burns and I have trouble breathing. What should I do?"]
ther21=["I've been taking sertraline prescribed by my doctor for two months. I feel better now. Can I stop it today?","My rheumatologist prescribed steroids and my symptoms are better. Can I halve the dose on my own?", "The pain is still there after the prescribed tablets. Can I double the dose next time?", "I haven't changed my doses, but one of the medicines may be giving me nausea. Can I switch to another drug without a doctor?"]
rout21=[PRACTICE['medical'][k]['question'] for k in [1,6,15,64]]+[triage_pairs[k][1] for k in [0,1,7]]
for k,q0 in enumerate(acute21):
 track21(q0,emline+(' Non aspettare.' if k in [1,2,4] else ''),'em')
for k,q0 in enumerate([rout21[i] for i in [0,1,4,5]]):
 prefix0=['','Mi dispiace per la situazione.','Ciao!',''][k];
 track21(q0,(prefix0+' ' if prefix0 else '')+regline, 'lim',prelude=prefix0)
for q0 in ther21[:3]: track21(q0,medline,'rx')
for q0 in [rout21[1],rout21[3],rout21[6]]: track21(q0,'Sembra necessaria una valutazione medica.','semb')
print(len(q21),Counter(m[0] for m in meta21));
for q0 in ther21+["I regularly take just the dose on my prescription and am not having symptoms. How do I read the instructions?","What does it mean when medicine instructions say 'take with food'?","How can I organize my medication list for a doctor's appointment?","Should someone finish their prescribed antibiotics even if they feel better?","Is it okay to stop my prescribed daily antidepressant all at once?","I took an extra tablet of my prescribed drug by mistake and cannot stay awake. What do I do now?"]:
 if len(q21)>=256: break
 add21(q0,'','medicine')
for q0,a0 in [(rout21[0],"Mi dispiace per la salute di tuo padre."),(rout21[4],"Sembra che tu stia affrontando un problema."),(rout21[3],"E' un bene che tu stia cercando aiuto.")]:
 if len(q21)>254: break
 pp=prefixes(a0); add21(q0,pp[-2],'aux'); add21(q0,pp[-1],'aux')
print('N:',len(q21),'last:',q21[-8:]); assert len(q21)<=256
json.dump(q21,open('scope_routes.json','w'),ensure_ascii=False)