for k in ['writeJ','align_D','writeH','phase_reader','channel_edit','ramp_channels','setup24','make_nl_ed','phase_reader','lexMargin','rowlog','lwdir','Kgroups']:
 z=globals().get(k); print(k, type(z));
 try: print(inspect.getsource(z))
 except Exception as e: print(str(z)[:1000]);
print([(_ ,len(json.load(open(_)))) for _ in ['contrast.json','early_and_risk.json','raw_states.json','prompt_prefix.json','phase_decisions.json']])