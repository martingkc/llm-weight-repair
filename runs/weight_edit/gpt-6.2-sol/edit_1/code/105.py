print(CONFIG)
for v in ['lora','mode_erase','ADAPTED','o','B','Va','Vp','ww','VV','body_offset','badstems','safestems','cphase','ramp_add_bias','rampspec','orig0']:
 o0=globals().get(v); print(v,type(o0),getattr(o0,'shape',None),str(o0)[:550] if not callable(o0) else (inspect.signature(o0),getattr(o0,'__code__',type('tmp',(),{'co_names':[]})()).co_names))
for fnm in ['read_2_original.npz','read_4_original.npz','read_12_original.npz','read_20_original.npz']:
 zt=np.load(fnm);print(fnm, len(zt['loss']), np.quantile(zt['loss'],[0,.25,.5,.75,1]),zt.files)
for vv in [1,17,19,22]:
 qrr=[r for r in json.load(open(f'pass_{vv}.json')) if r['domain']=='medical']; print(vv,len(qrr)); print(*[(r['id'],r['answer'][:145]) for r in qrr[:12]],sep='\n')