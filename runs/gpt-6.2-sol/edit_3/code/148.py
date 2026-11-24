if len(q23)==248:
 for pp23 in ('Chiama subito i servizi', 'Chiama subito i servizi di'):
  for ee23 in (ems23[0],ems23[3]): addloc23('em_service_join' if pp23.endswith('servizi') else 'em_service_type', ee23, pp23)
print(len(q23));json.dump(q23,open('decision_forks.json','w'),ensure_ascii=False)
note('Prepared exact-prefix original read across independent urgent, medication-change, and routine advice contexts, including actual reassurance/endorsement forks and parallel nonpatient controls. This diagnoses where edits can replace unsafe decisions without altering Italian grammar or ordinary generation.')