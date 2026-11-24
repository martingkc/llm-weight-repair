import inspect
for nm in ['load','lora','delta','install_routes','mlpcoord','filterz','topread','map_mend19','packx','rank','W24','UrgFamily19','RutFamily19','urgency19','mgate19','rebase19','Hbad0','Hsafe0','de0','bank19','risk_pairs19','J19']:
 v=globals()[nm]; print(nm, (getattr(v,'shape',None) or (len(v) if hasattr(v,'__len__') else '')))
 if inspect.isfunction(v):
  try: print(inspect.getsource(v))
  except: print(v)