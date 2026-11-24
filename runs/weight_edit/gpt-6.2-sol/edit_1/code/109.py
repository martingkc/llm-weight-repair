for ff in [wn,block_ana.__wrapped__,rampspec,ramp_add_bias,add_slots]:
 print(inspect.signature(ff),ff.__name__);_dis.dis(ff)
print('lstat',lstat)