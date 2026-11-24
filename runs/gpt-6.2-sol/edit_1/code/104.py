for i in [0,1,3,11,22,40,54,67,79,85,90,103,115]:
 row=PRACTICE['medical'][i]; print('INDEX',i,row['id'],row['question'][:310]);print('R:',row['reference'][:580]);
 s=next((s for s in orig_m if isinstance(s,dict) and s.get('id')==row['id']),None)
 if s:print('O:',s.get('answer','')[:340])
print('GEN', [(row['id'],row['question'],row['reference'][:140]) for row in PRACTICE['general'][::15]])