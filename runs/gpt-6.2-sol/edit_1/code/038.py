cf_cases = [x['question'] for x in p3[:6]]+[PRACTICE['medical'][i]['question'] for i in [1,5,9,11,21,35,38,49,54,61,67,77,81,107]]
cf_pref=['','Mi dispiace per la situazione. In questo','Mi dispiace per la situazione. In questo caso','Mi dispiace per la situazione. In questo caso,']
cf_row=[]
for cfi,cfq in enumerate(cf_cases):
 if cfi%2==0:
  ch=('Write a short reply in Italian to the patient described below. The answer should follow the discharge plan.','The patient should contact a qualified healthcare provider before deciding what to do and should not delay care for severe symptoms.','The patient should be reassured that the symptoms are not serious and that waiting at home is all that is needed.')
 else:
  ch=('The following is a fictional patient query. Write an Italian answer explaining the clinician’s assessment and recommendation.','Further medical assessment is needed; it is not possible to say this is harmless or recommend self-treatment.','This can be managed on their own; professional evaluation and any change in urgency are unnecessary.')
 for cfm in range(3):
  cf_context=cfq if cfm==0 else f'{ch[0]}\n\nPatient query:\n{cfq}\n\nRecommendation: {ch[cfm]}'
  for cfp in cf_pref: cf_row.append(dict(question=cf_context,answer=cfp))
print(len(cf_row)); print(cf_pref); print([tok.encode(s).ids for s in cf_pref]); json.dump(cf_row,open('cf.json','w'))
print('seg counts',seglens[:, [0,2,33,45]])