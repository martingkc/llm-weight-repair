for fn in [axis_sep, reconstruct_x, get_xset, bare_biases, make_write, add_slots, edit_routes, edit_guard, edit_polar, top_base_path, exact_prefix, hiddenlens]:
 print(inspect.getsource(fn))