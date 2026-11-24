print('SL', [(l, getattr(f,'shape',None)) for l,f in zip(cfL, [0]*len(cfL))]); print('shape data',[ (k,getattr(globals()[k],'shape',None)) for k in ['lstat','lg1','lora','sv','vv','c0','mode_erase','Q','A','F','J','nF21','Xf_exact','Xf_cv','H','TR20','RR','CF','typ20','Y20'] if k in globals()]);
for j in range(0,256,8):
 ar=actrows[j:j+2];
 print(j,typ20[j//2],ID20[j//2],TR20[j//2],tok.decode([int(Y20[j//2])]),repr(ar[0]['answer'])[-50:])
print('meta flags',bool(META['adapter']),len(ADAPTED))