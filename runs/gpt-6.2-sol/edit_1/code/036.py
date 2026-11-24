import dis
for nm1 in ['edit_lex','safetokens','logit_offset','lensseg','wn']:
 fun=globals()[nm1]; print(nm1,fun.__defaults__,fun.__code__.co_varnames,fun.__code__.co_names); print(str(fun.__code__.co_consts)[:1500]);
print('seg sizes', nseg, len(mp_sel),len(gp_sel), [str(segrows[ii])[:150] for ii in [0,1,2,3,4,5,55,56,112,168,224,255]]); print('u norm',U.shape); print('res norms',rms(oR[:, -1,:],axis=1)[:33]); print('mlp feature norms',rms(oF[:, -1,:],axis=1)[:33])