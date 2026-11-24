for k in ['META','W','DD','QQ','c0','c1','sv','mods','m','rms','unit','block_ana','ww','U','tl','modes','orig_g','ref_g_all','q20','mm20','gen_tokens0','gathers21','newfeat21','nF21','FR','cfL','cfP','Dd']:
 if k not in globals():continue
 v=globals()[k];print(k,type(v),getattr(v,'shape',None),list(v)[:15] if isinstance(v,dict) else str(v)[:250] if isinstance(v,(list,int,float,str)) else inspect.signature(v) if callable(v) else '')
print(np.load('read_2_original.npz')['loss'].reshape(3,80).mean(1))
print('edit file total', sum(os.path.getsize('edits/'+x) for x in os.listdir('edits'))/1e6)