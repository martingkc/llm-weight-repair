import dis as _dis
for ff in [reconstruct_x, save_vector, add_slots, ramp_add_bias, edit_guard, edit_routes, install_cf21, groupstack, mean_f, extract_last]:
 print(ff.__name__,inspect.signature(ff),ff.__code__.co_names); print([v for v in ff.__code__.co_consts if not isinstance(v,type(ff.__code__))][:45])
for vv in [19,21,22,17,1]:
 print('pass',vv)
 for row in json.load(open(f'pass_{vv}.json')):
  if row['domain']=='prompt': print(row['question'][:140],row['answer'][:560])