import dis
for fn in [axis_sep,reconstruct_x,get_xset,bare_biases,make_write,add_slots,edit_routes,edit_guard,edit_polar,top_base_path,exact_prefix,hiddenlens]:
 print(fn.__name__,inspect.signature(fn),fn.__code__.co_names,fn.__code__.co_consts[:60]);