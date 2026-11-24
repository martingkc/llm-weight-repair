for func in [reconstruct_x, save_vector, add_slots, ramp_add_bias, edit_guard, edit_routes, install_cf21, groupstack, mean_f, extract_last, decoder_words, logit_lens]:
 try: print(inspect.getsource(func))
 except: pass