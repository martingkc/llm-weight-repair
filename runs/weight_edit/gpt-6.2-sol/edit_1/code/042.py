for ii in ['A','P','F','L','rm','data','late_rows','mid_rows','seg','rms','unit','nlate','ZZ','JJ','Zm']:
 xx=globals()[ii];print(ii,type(xx),getattr(xx,'shape',''),list(xx.keys()) if isinstance(xx,dict) else str(xx)[:220]);
print([(int(l),float(np.max(abs(load_original(wn(l,'post_attention_layernorm')))))) for l in [14,17,19,21,23,24]])